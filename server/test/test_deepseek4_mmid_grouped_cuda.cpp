#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cuda.h"
#include "ggml.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <unistd.h>

static bool run_case(
        ggml_backend_t backend,
        ggml_type type,
        int width,
        bool write_output,
        std::ofstream & output) {
    constexpr int k_dim = 256;
    constexpr int n_rows = 128;
    constexpr int n_experts = 32;
    constexpr int top_k = 8;

    ggml_init_params params = {16 * 1024 * 1024, nullptr, true};
    ggml_context * ctx = ggml_init(params);
    if (ctx == nullptr) {
        return false;
    }

    ggml_tensor * weights = ggml_new_tensor_3d(ctx, type, k_dim, n_rows, n_experts);
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k_dim, 1, width);
    ggml_tensor * ids = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, top_k, width);
    ggml_set_input(weights);
    ggml_set_input(input);
    ggml_set_input(ids);

    ggml_tensor * result = ggml_mul_mat_id(ctx, weights, input, ids);
    ggml_set_output(result);
    ggml_cgraph * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, result);

    ggml_gallocr_t alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    if (!ggml_gallocr_alloc_graph(alloc, graph)) {
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    std::mt19937 rng(20260713u + (unsigned) type * 97u + (unsigned) width);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> weights_f((size_t) k_dim * n_rows * n_experts);
    std::vector<float> input_f((size_t) k_dim * width);
    for (float & value : weights_f) {
        value = dist(rng);
    }
    for (float & value : input_f) {
        value = dist(rng);
    }

    std::vector<uint8_t> weights_q(ggml_nbytes(weights));
    const size_t quantized = ggml_quantize_chunk(
        type, weights_f.data(), weights_q.data(), 0, n_rows * n_experts, k_dim, nullptr);
    if (quantized != weights_q.size()) {
        std::fprintf(stderr, "quantize size mismatch type=%s got=%zu expected=%zu\n",
                     ggml_type_name(type), quantized, weights_q.size());
        ggml_gallocr_free(alloc);
        ggml_free(ctx);
        return false;
    }

    std::vector<int32_t> ids_h((size_t) top_k * width);
    for (int token = 0; token < width; ++token) {
        for (int slot = 0; slot < top_k; ++slot) {
            ids_h[(size_t) token * top_k + slot] = (token * 3 + slot * 5) % n_experts;
        }
    }

    ggml_backend_tensor_set(weights, weights_q.data(), 0, weights_q.size());
    ggml_backend_tensor_set(input, input_f.data(), 0, input_f.size() * sizeof(float));
    ggml_backend_tensor_set(ids, ids_h.data(), 0, ids_h.size() * sizeof(int32_t));

    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    std::vector<float> result_h(ggml_nelements(result));
    if (status == GGML_STATUS_SUCCESS) {
        ggml_backend_tensor_get(result, result_h.data(), 0, result_h.size() * sizeof(float));
        if (write_output) {
            output.write(reinterpret_cast<const char *>(result_h.data()),
                         (std::streamsize) (result_h.size() * sizeof(float)));
        }
    }

    std::printf("[mmid-grouped-test] type=%s width=%d pairs=%d status=%d bytes=%zu\n",
                ggml_type_name(type), width, width * top_k, (int) status,
                result_h.size() * sizeof(float));
    ggml_gallocr_free(alloc);
    ggml_free(ctx);
    return status == GGML_STATUS_SUCCESS && output.good();
}

static bool baseline_uses_mmvq(ggml_type type, int width) {
    if (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q6_K || type == GGML_TYPE_Q5_K) {
        return width <= 4;
    }
    return width <= 8;
}

static int run_child(const char * mode, const char * output_path) {
    ggml_backend_t backend = ggml_backend_cuda_init(0);
    if (backend == nullptr) {
        std::fprintf(stderr, "GPU backend unavailable\n");
        return 1;
    }

    std::ofstream output(output_path, std::ios::binary | std::ios::trunc);
    const ggml_type types[] = {
        GGML_TYPE_Q4_K, GGML_TYPE_Q6_K, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0, GGML_TYPE_Q5_K,
    };
    const int widths[] = {2, 4, 8, 9, 16};
    const bool grouped = std::strcmp(mode, "grouped") == 0;
    if (!grouped && std::strcmp(mode, "legacy") != 0) {
        return 2;
    }
    bool ok = output.good();
    for (ggml_type type : types) {
        for (int width : widths) {
            const bool baseline_mmvq = baseline_uses_mmvq(type, width);
            if (!grouped && !baseline_mmvq) {
                continue;
            }
            ok = run_case(backend, type, width, baseline_mmvq, output) && ok;
        }
    }
    output.close();
    ggml_backend_free(backend);
    return ok ? 0 : 1;
}

static std::vector<char> read_file(const char * path) {
    std::ifstream input(path, std::ios::binary | std::ios::ate);
    if (!input) {
        return {};
    }
    const std::streamsize size = input.tellg();
    input.seekg(0);
    std::vector<char> data((size_t) size);
    input.read(data.data(), size);
    return data;
}

static size_t count_records(const std::vector<char> & data, const char * needle) {
    const std::string text(data.begin(), data.end());
    size_t count = 0;
    size_t offset = 0;
    while ((offset = text.find(needle, offset)) != std::string::npos) {
        ++count;
        offset += std::strlen(needle);
    }
    return count;
}

int main(int argc, char ** argv) {
    if (argc == 4 && std::strcmp(argv[1], "--child") == 0) {
        return run_child(argv[2], argv[3]);
    }
    if (argc != 1) {
        std::fprintf(stderr, "usage: %s [--child legacy|grouped OUTPUT]\n", argv[0]);
        return 2;
    }

    const std::string prefix = "mmid_grouped_" + std::to_string((long long) getpid());
    const std::string legacy_path = prefix + "_legacy.bin";
    const std::string grouped_path = prefix + "_enabled.bin";
    const std::string legacy_log_path = prefix + "_legacy.log";
    const std::string grouped_log_path = prefix + "_enabled.log";
    const std::string executable = argv[0];
    const std::string legacy_cmd =
        "DFLASH_MMID_TELEMETRY=1 DFLASH_MMID_GROUPED=0 '" + executable +
        "' --child legacy " + legacy_path + " 2>" + legacy_log_path;
    const std::string grouped_cmd =
        "DFLASH_MMID_TELEMETRY=1 DFLASH_MMID_GROUPED=1 '" + executable +
        "' --child grouped " + grouped_path + " 2>" + grouped_log_path;

    const int legacy_status = std::system(legacy_cmd.c_str());
    const int grouped_status = std::system(grouped_cmd.c_str());
    const std::vector<char> legacy = read_file(legacy_path.c_str());
    const std::vector<char> grouped = read_file(grouped_path.c_str());
    const std::vector<char> legacy_log = read_file(legacy_log_path.c_str());
    const std::vector<char> grouped_log = read_file(grouped_log_path.c_str());
    const size_t legacy_grouped = count_records(legacy_log, "variant=grouped");
    const size_t legacy_legacy = count_records(legacy_log, "variant=legacy_moe");
    const size_t grouped_grouped = count_records(grouped_log, "variant=grouped");
    const size_t grouped_legacy = count_records(grouped_log, "variant=legacy_moe");
    const bool output_parity = !legacy.empty() && legacy == grouped;
    const bool pass = legacy_status == 0 && grouped_status == 0 &&
        output_parity && legacy_grouped == 0 && legacy_legacy == 12 &&
        grouped_grouped == 25 && grouped_legacy == 0;
    if (pass) {
        std::remove(legacy_path.c_str());
        std::remove(grouped_path.c_str());
        std::remove(legacy_log_path.c_str());
        std::remove(grouped_log_path.c_str());
    }
    std::printf("[mmid-grouped-test] legacy_status=%d grouped_status=%d bytes=%zu "
                "legacy_grouped=%zu legacy_legacy=%zu grouped_grouped=%zu grouped_legacy=%zu "
                "parity=%s\n",
                legacy_status, grouped_status, legacy.size(), legacy_grouped, legacy_legacy,
                grouped_grouped, grouped_legacy, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
