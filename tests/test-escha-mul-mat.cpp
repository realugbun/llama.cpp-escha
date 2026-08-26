// Checks ggml_escha_mul_mat against numpy references built by dump_op_case_dense.py.
//
// The reference path folds the projection to a dense matrix and does a plain matmul,
// so it shares no code with the op, which decodes tiles on the fly and rotates the
// activations instead. Agreement means the fused form is right, not just repeatable.
//
// Every case runs on the CPU and, when one is present, on the GPU. Both are scored
// against the same reference rather than against each other, so a shared bug in the
// two ggml paths cannot hide.
//
// The cases are carved out of the real dense checkpoint: `dep` has to hold valid bit
// indices, so test-backend-ops cannot cover this op with random data.
//
//   usage: test-escha-mul-mat [path-to-escha-mul-mat-cases.gguf] [--bench]

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static const char * DEFAULT_CASES =
    "/media/aj-homeserver/windows/escha-gguf-dense/escha-mul-mat-cases.gguf";

struct escha_case {
    ggml_tensor * code;
    ggml_tensor * rin;
    ggml_tensor * rout;
    ggml_tensor * lut;
    ggml_tensor * dep;
    ggml_tensor * x;
    ggml_tensor * yref;
};

static ggml_tensor * need(ggml_context * ctx, const std::string & name) {
    ggml_tensor * t = ggml_get_tensor(ctx, name.c_str());
    if (!t) {
        fprintf(stderr, "missing tensor '%s' in the case file\n", name.c_str());
        exit(1);
    }
    return t;
}

// run one case on `backend`, or on the CPU inline when backend is null
static std::vector<float> run_case(const escha_case & c, ggml_backend_t backend) {
    const int64_t n = ggml_nelements(c.yref);

    ggml_init_params ip = {
        /*.mem_size   =*/ ggml_tensor_overhead()*32 + ggml_graph_overhead() + (backend ? 0 : 64u*1024*1024),
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ backend != nullptr,
    };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * src[6] = { c.code, c.rin, c.rout, c.lut, c.dep, c.x };
    ggml_tensor * dev[6];
    for (int i = 0; i < 6; ++i) {
        dev[i] = backend ? ggml_dup_tensor(ctx, src[i]) : src[i];
    }

    ggml_tensor * y = ggml_escha_mul_mat(ctx, dev[0], dev[1], dev[2], dev[3], dev[4], dev[5]);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    std::vector<float> out(n);

    if (backend) {
        ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
        if (!buf) {
            fprintf(stderr, "failed to allocate backend buffer\n");
            exit(1);
        }
        for (int i = 0; i < 6; ++i) {
            ggml_backend_tensor_set(dev[i], src[i]->data, 0, ggml_nbytes(src[i]));
        }
        if (ggml_backend_graph_compute(backend, gf) != GGML_STATUS_SUCCESS) {
            fprintf(stderr, "backend compute failed\n");
            exit(1);
        }
        ggml_backend_tensor_get(y, out.data(), 0, n*sizeof(float));
        ggml_backend_buffer_free(buf);
    } else {
        ggml_graph_compute_with_ctx(ctx, gf, 12);
        std::copy((const float *) y->data, (const float *) y->data + n, out.begin());
    }

    ggml_free(ctx);
    return out;
}

// The op computes codebook A rather than reading the table, so the table it is handed
// has to BE codebook A. A checkpoint on one of escha's other codebooks would otherwise
// decode to plausible-looking garbage.
static bool check_codebook(const ggml_tensor * lut) {
    if (ggml_nelements(lut) != 65536 || lut->type != GGML_TYPE_F16) {
        printf("lut is not (65536,) f16\n");
        return false;
    }

    const ggml_fp16_t * t = (const ggml_fp16_t *) lut->data;
    int64_t bad = 0;

    for (uint32_t i = 0; i < 65536; ++i) {
        const uint32_t x = ((i*0xcbac1fedu) & 0x8fff8fffu) ^ 0x3b603b60u;
        ggml_fp16_t lo, hi;
        memcpy(&lo, (const char *) &x,     sizeof(lo));
        memcpy(&hi, (const char *) &x + 2, sizeof(hi));

        const float v = ggml_fp16_to_fp32(lo) + ggml_fp16_to_fp32(hi);
        if (ggml_fp16_to_fp32(ggml_fp32_to_fp16(v)) != ggml_fp16_to_fp32(t[i])) {
            bad++;
        }
    }

    printf("codebook: closed form vs shipped table  %s (%" PRId64 " of 65536 differ)\n\n",
           bad ? "FAIL" : "exact", bad);
    return bad == 0;
}

// max abs difference and relative RMS against the reference
// Prefill runs on tensor cores, which requires fp16 activations -- the same thing escha's own
// runtime does. That is a real, expected ~2.1e-4 and not a defect, so the prefill cases get a
// tolerance that admits it while still catching a broken kernel by orders of magnitude. The
// weights stay exact either way, and batch 1 (rows <= ESCHA_GEN_MAX_ROWS) never leaves fp32.
// ESCHA_NO_MMA=1 selects the scalar fp32 path in the op, and tightens this back to 1e-5.
static double tolerance_for(int64_t rows, bool cuda) {
    static const bool mma = getenv("ESCHA_NO_MMA") == nullptr;
    return (cuda && mma && rows > 16) ? 5e-4 : 1e-5;
}

static bool score(const char * label, uint32_t ic, int64_t K, int64_t IC, int64_t OC,
                  int64_t rows, const std::vector<float> & got, const ggml_tensor * yref) {
    const float * b = (const float *) yref->data;
    const int64_t n = ggml_nelements(yref);

    double max_abs = 0.0, sse = 0.0, ssr = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = (double) got[i] - (double) b[i];
        max_abs = std::max(max_abs, std::fabs(d));
        sse += d*d;
        ssr += (double) b[i] * (double) b[i];
    }
    const double rel = std::sqrt(sse/ssr);

    // f32 reassociation only on the fp32 paths: the op runs a butterfly Hadamard and
    // accumulates per tile, the reference multiplies by a dense folded matrix
    const double tol = tolerance_for(rows, strcmp(label, "cuda") == 0);
    const bool ok = rel < tol;

    printf("case %u  %-4s  K=%" PRId64 "  %4" PRId64 " -> %-4" PRId64 "  rows=%-4" PRId64
           "  max_abs=%.3e  rel_rms=%.3e  (tol %.0e)  %s\n",
           ic, label, K, IC, OC, rows, max_abs, rel, tol, ok ? "OK" : "FAIL");
    return ok;
}

// wall-clock per op, so kernel changes can be judged without a model load
static void bench(const escha_case & c, ggml_backend_t backend, uint32_t ic) {
    const int64_t IC   = c.code->ne[2]*16;
    const int64_t OC   = c.code->ne[1]*16;
    const int64_t rows = c.x->ne[1];
    const int64_t decodes = rows*IC*OC;

    ggml_init_params ip = { ggml_tensor_overhead()*32 + ggml_graph_overhead(), nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    ggml_tensor * src[6] = { c.code, c.rin, c.rout, c.lut, c.dep, c.x };
    ggml_tensor * dev[6];
    for (int i = 0; i < 6; ++i) {
        dev[i] = ggml_dup_tensor(ctx, src[i]);
    }
    ggml_tensor * y = ggml_escha_mul_mat(ctx, dev[0], dev[1], dev[2], dev[3], dev[4], dev[5]);

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, y);

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, backend);
    for (int i = 0; i < 6; ++i) {
        ggml_backend_tensor_set(dev[i], src[i]->data, 0, ggml_nbytes(src[i]));
    }

    for (int i = 0; i < 8; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);

    const int reps = 200;
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < reps; ++i) {
        ggml_backend_graph_compute(backend, gf);
    }
    ggml_backend_synchronize(backend);
    const double us = (double)(ggml_time_us() - t0)/reps;

    printf("bench %u  K=%" PRId64 "  %4" PRId64 " -> %-4" PRId64 "  rows=%-4" PRId64
           "  %7.1f us/op   %6.1f Gdecode/s\n",
           ic, c.code->ne[0]/16, IC, OC, rows, us, decodes/us*1e-3);

    ggml_backend_buffer_free(buf);
    ggml_free(ctx);
}

int main(int argc, char ** argv) {
    const char * path = argc > 1 ? argv[1] : DEFAULT_CASES;
    const bool do_bench = argc > 2 && std::string(argv[2]) == "--bench";

    ggml_context * ctx = nullptr;
    gguf_init_params gp = { /*.no_alloc =*/ false, /*.ctx =*/ &ctx };
    gguf_context * gguf = gguf_init_from_file(path, gp);
    if (!gguf) {
        fprintf(stderr, "failed to open case file '%s'\n", path);
        fprintf(stderr, "generate it with dump_op_case_dense.py\n");
        return 1;
    }

    const int key = gguf_find_key(gguf, "escha.n_cases");
    const uint32_t n_cases = key >= 0 ? gguf_get_val_u32(gguf, key) : 0;
    if (n_cases == 0) {
        fprintf(stderr, "case file declares no cases\n");
        return 1;
    }

    ggml_tensor * lut    = need(ctx, "escha_lut");
    ggml_tensor * dep_k2 = need(ctx, "escha_dep_k2");
    ggml_tensor * dep_k3 = need(ctx, "escha_dep_k3");

    bool ok = check_codebook(lut);

    ggml_backend_t gpu = nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_GPU) {
            gpu = ggml_backend_dev_init(dev, nullptr);
            printf("gpu backend: %s\n\n", ggml_backend_name(gpu));
            break;
        }
    }
    if (!gpu) {
        printf("no gpu backend found, scoring the cpu path only\n\n");
    }

    for (uint32_t i = 0; i < n_cases; ++i) {
        const std::string p = "case" + std::to_string(i);

        escha_case c;
        c.code = need(ctx, p + ".code");
        c.rin  = need(ctx, p + ".rin");
        c.rout = need(ctx, p + ".rout");
        c.x    = need(ctx, p + ".x");
        c.yref = need(ctx, p + ".yref");
        c.lut  = lut;

        const int64_t K = c.code->ne[0]/16;
        c.dep = K == 2 ? dep_k2 : dep_k3;

        const int64_t IC   = c.code->ne[2]*16;
        const int64_t OC   = c.code->ne[1]*16;
        const int64_t rows = c.x->ne[1];

        ok &= score("cpu", i, K, IC, OC, rows, run_case(c, nullptr), c.yref);
        if (gpu) {
            ok &= score("cuda", i, K, IC, OC, rows, run_case(c, gpu), c.yref);
        }
    }

    if (do_bench && gpu) {
        printf("\n");
        for (uint32_t i = 0; i < n_cases; ++i) {
            const std::string p = "case" + std::to_string(i);
            escha_case c;
            c.code = need(ctx, p + ".code");
            c.rin  = need(ctx, p + ".rin");
            c.rout = need(ctx, p + ".rout");
            c.x    = need(ctx, p + ".x");
            c.yref = need(ctx, p + ".yref");
            c.lut  = lut;
            c.dep  = c.code->ne[0]/16 == 2 ? dep_k2 : dep_k3;
            bench(c, gpu, i);
        }
    }

    if (gpu) {
        ggml_backend_free(gpu);
    }
    gguf_free(gguf);
    ggml_free(ctx);

    printf("\n%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
