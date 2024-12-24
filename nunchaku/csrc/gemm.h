#pragma once

#include "interop/torch.h"
#include "Serialization.h"
#include "Linear.h"
#include "debug.h"

#include "kernels/gemm_w4a4.h"
#include "kernels/awq/gemv_awq.h"

class QuantizedGEMM { // : public torch::CustomClassHolder {
public:
    void init(int64_t in_features, int64_t out_features, bool bias, bool bf16, int8_t deviceId) {
        spdlog::info("Initializing QuantizedGEMM");
        
        size_t val = 0;
        checkCUDA(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
        checkCUDA(cudaDeviceGetLimit(&val, cudaLimitStackSize));
        spdlog::debug("Stack={}", val);

        net = std::make_unique<GEMM_W4A4>((int)in_features, (int)out_features, bias, bf16 ? Tensor::BF16 : Tensor::FP16, Device::cuda((int)deviceId));
    }

    void reset() {
        debugContext.reset();
        net.reset();
        Tensor::synchronizeDevice();
    }

    void load(std::string path) {
        checkModel();

        spdlog::info("Loading weights from {}", path);
        std::shared_ptr<SafeTensors> provider = std::make_shared<SafeTensors>(path);
        net->loadParams(*provider);
        Tensor::synchronizeDevice();
    }

    void loadParam(std::string key, torch::Tensor src) {
        checkModel();

        net->loadParam(key, *(net->params[key].tensor), from_torch(src));
        Tensor::synchronizeDevice();
    }

    torch::Tensor forward(torch::Tensor x) {
        checkModel();

        spdlog::info("QuantizedGEMM forward");

        x = x.contiguous();

        Tensor result = std::get<Tensor>(net->forward(
            from_torch(x)
        ));
        
        torch::Tensor output = to_torch(result);
        Tensor::synchronizeDevice();

        return output;
    }

    std::string dumpTensorBF16(Tensor x) {
        std::stringstream ss;
        for (int i = 0; i < 256; i++) {
            ss << spdlog::fmt_lib::format("{:.3f} ", (float)(x.data_ptr<__nv_bfloat16>()[i]));
        }
        ss << std::endl;
        return ss.str();
    }

    std::string dumpTensorINT4(Tensor x) {
        using spdlog::fmt_lib::format;

        const int M = x.shape[0];
        const int K = x.shape[1] * 2;
        
        assert(x.dtype() == Tensor::INT8);

        // activation: row major, [M / BLOCK_M, K / WARP_K, NUM_WARPS, WARP_M_TILES, WARP_SIZE] of packed_act_t (uint4)

        constexpr int BLOCK_M = 256;
        constexpr int WARP_K = 64;
        constexpr int NUM_WARPS = 8;
        constexpr int WARP_M_TILES = 2;
        constexpr int WARP_SIZE = 32;

        std::stringstream ss;
        for (int bm = 0; bm < M / BLOCK_M; bm++) {
            for (int bn = 0; bn < K / WARP_K; bn++) {
                for (int warpId = 0; warpId < NUM_WARPS; warpId++) {
                    ss << format("[bm={},bn={},warp={}] ", bm, bn, warpId);
                    const int offset = ((bm * (K / WARP_K) + bn) * NUM_WARPS + warpId) * WARP_M_TILES * WARP_SIZE * 4;

                    for (int i = 0; i < 16; i++) {
                        assert(offset + i < x.numel() / 4);
                        uint32_t val = x.data_ptr<uint32_t>()[offset + i];
                        ss << "{";
                        for (int j = 0; j < 8; j++) {
                            int i4val = (val >> (j * 4)) & 0xf;
                            if (i4val & 0x8) {
                                i4val = -((~i4val & 0x7) + 1);
                            }
                            ss << format("{} ", i4val);
                        }
                        ss << format("}} {:x} ", val);
                    }
                    ss << std::endl;
                }
            }
        }
        
        ss << std::endl;
        return ss.str();
    }

    void quantize(torch::Tensor x) {
        checkModel();

        spdlog::debug("QuantizedGEMM quantize");

        x = x.contiguous();

        auto qout = net->quantize(
            from_torch(x)
        );
        
        Tensor act = qout.act.copy(Device::cpu());
        Tensor ascales = qout.ascales.copy(Device::cpu());
        Tensor lora_act = qout.lora_act.copy(Device::cpu());

        Tensor::synchronizeDevice();

        spdlog::debug("act = {}", dumpTensorINT4(act));
        spdlog::debug("ascales = {}", dumpTensorBF16(ascales));
    }

    
    void gemm(
        c10::optional<torch::Tensor> act,          // packed act [M, K / 2]
        c10::optional<torch::Tensor> wgt,          // packed act [N, K / 2]
        c10::optional<torch::Tensor> out,          // linear     [M, N]
        c10::optional<torch::Tensor> qout,         // packed act [M, N / 2]
        c10::optional<torch::Tensor> ascales,      // packed as  [K / 64, M]
        c10::optional<torch::Tensor> wscales,      // packed ws  [K / 64, N]
        c10::optional<torch::Tensor> oscales,      // packed as  [N / 64, M]
        c10::optional<torch::Tensor> poolout,      // linear     [M / PoolSize, N]
        c10::optional<torch::Tensor> lora_act_in,  // packed lora_act [M, R]
        c10::optional<torch::Tensor> lora_up,      // packed lora_wgt [N, R]
        c10::optional<torch::Tensor> lora_down,    // packed lora_wgt [N, R]
        c10::optional<torch::Tensor> lora_act_out, // packed lora_act [M, R]
        c10::optional<torch::Tensor> norm_q,       // linear     [HEAD_DIM]
        c10::optional<torch::Tensor> norm_k,       // linear     [HEAD_DIM]
        c10::optional<torch::Tensor> rotary_emb,   // linear     [M, HEAD_DIM / 2, 2, 2]
        c10::optional<torch::Tensor> bias,         // packed ws  [N]
        c10::optional<torch::Tensor> smooth_factor, // packed ws  [N], for quantization of the next layer
        bool act_unsigned,
        std::vector<float> lora_scales
    ) {
        std::cerr << "running gemm_w4a4: " << std::endl;

        auto getTensor = [](c10::optional<torch::Tensor> &t) {
            Tensor ret = t.has_value() ? from_torch(t.value()) : Tensor{};
            if (ret.valid()) {
                std::cerr << "  " << ret.shape.str() << std::endl;
            } else {
                std::cerr << "  <invalid>" << std::endl;
            }
            return ret;
        };
        gemm_w4a4(
            getTensor(act          ),
            getTensor(wgt          ),
            getTensor(out          ),
            getTensor(qout         ),
            getTensor(ascales      ),
            getTensor(wscales      ),
            getTensor(oscales      ),
            getTensor(poolout      ),
            getTensor(lora_act_in  ),
            getTensor(lora_up      ),
            getTensor(lora_down    ),
            getTensor(lora_act_out ),
            getTensor(norm_q       ),
            getTensor(norm_k       ),
            getTensor(rotary_emb   ),
            getTensor(bias         ),
            getTensor(smooth_factor),
            act_unsigned,
            lora_scales
        );
        Tensor::synchronizeDevice();
    }

    torch::Tensor gemv_awq(
        torch::Tensor _in_feats,
        torch::Tensor _kernel,
        torch::Tensor _scaling_factors,
        torch::Tensor _zeros,
        int64_t m,
        int64_t n,
        int64_t k,
        int64_t group_size)
    {
        Tensor result = ::gemv_awq(
            from_torch(_in_feats.contiguous()),
            from_torch(_kernel.contiguous()),
            from_torch(_scaling_factors.contiguous()),
            from_torch(_zeros.contiguous()),
            (int)m, 
            (int)n, 
            (int)k, 
            (int)group_size
        );

        torch::Tensor output = to_torch(result);
        Tensor::synchronizeDevice();

        return output;
    }

    void startDebug() {
        debugContext = std::make_unique<DebugContext>();
    }
    void stopDebug() {
        debugContext.reset();
    }

    auto getDebugResults() {
        // c10::Dict<std::string, torch::Tensor> result;
        std::map<std::string, torch::Tensor> result;


        if (debugContext) {
            for (auto &&[key, value] : debugContext->tensors) {
                // result.insert(key, to_torch(value));
                result[key] = to_torch(value);
            }
        }
        
        return result;
    }

private:
    void checkModel() {
        if (!net) {
            throw std::runtime_error("Model not initialized");
        }
    }

private:
    std::unique_ptr<GEMM_W4A4> net;
    std::unique_ptr<DebugContext> debugContext;
};

class QuantizedGEMMW4A4 { // : public torch::CustomClassHolder {
public:
    QuantizedGEMMW4A4() {
        spdlog::info("Initializing QuantizedGEMMW4A4");
        size_t val = 0;
        checkCUDA(cudaDeviceSetLimit(cudaLimitStackSize, 8192));
        checkCUDA(cudaDeviceGetLimit(&val, cudaLimitStackSize));
        spdlog::info("Stack size={}", val);
    }

    GEMM_W4A4::QuantizedActivation quantize(Tensor input, Tensor lora_down, Tensor smooth) {
        spdlog::info("QuantizedGEMM quantize");

        const int M = input.numel() / input.shape[-1];
        const int K = input.shape[-1];

        auto shape = TensorShape(input.shape.dataExtent);
        shape[-1] = K / 2;

        const int lora_rank = lora_down.shape[-1];

        spdlog::info("Quantize: M={}, K={}, lora_rank={}", M, K, lora_rank);
        
        GEMM_W4A4::QuantizedActivation qact;
        qact.act = Tensor::allocate(shape, Tensor::INT8, input.device());
        qact.ascales = Tensor::allocate({K / 64, M}, input.dtype(), input.device());    // group_size=64
        qact.lora_act = Tensor::allocate({M, lora_rank}, Tensor::FP32, input.device());
        qact.is_unsigned = false;

        quantize_w4a4_act_fuse_lora(input, qact.act, qact.ascales, lora_down, qact.lora_act, smooth);

        return qact;
    }

    void quantize_pack_w4a4_wgt(
        torch::Tensor input,
        torch::Tensor output,
        torch::Tensor oscales
    ) {
        spdlog::info("QuantizedGEMM quantize_w4a4_wgt");
        quantize_w4a4_wgt(from_torch(input), from_torch(output), from_torch(oscales));
        Tensor::synchronizeDevice();
    }
    
    torch::Tensor gemm(
        c10::optional<torch::Tensor> act,           // act [M, K]
        c10::optional<torch::Tensor> wgt,           // packed weight [N, K / 2]
        c10::optional<torch::Tensor> wscales,       // packed [K / 64, N]
        c10::optional<torch::Tensor> lora_up,       // packed [N, R]
        c10::optional<torch::Tensor> lora_down,     // packed [K, R]
        // c10::optional<torch::Tensor> bias,       // [N]
        c10::optional<torch::Tensor> smooth_factor, // [K], for quantization
        std::vector<float> lora_scales              // [R / 16]
    ) {
        auto getTensor = [](c10::optional<torch::Tensor> &t) {
            Tensor ret = t.has_value() ? from_torch(t.value().contiguous()) : Tensor{};
            if (ret.valid()) {
                std::cerr << "  " << ret.shape.str() << std::endl;
            } else {
                std::cerr << "  <invalid>" << std::endl;
            }
            return ret;
        };
        spdlog::debug("QuantizedGEMM act quantize");
        auto input = getTensor(act);
        auto smooth = getTensor(smooth_factor);
        auto _lora_down = getTensor(lora_down);
        
        const int M = input.numel() / input.shape[-1];
        const int K = input.shape[-1];
        const int lora_rank = 32;

        auto device = input.device();
        auto dtype = input.dtype();

        spdlog::info("Quantize: M={}, K={}, lora_rank={}", M, K, lora_rank);
        
        GEMM_W4A4::QuantizedActivation qact;

        auto qact_shape = TensorShape(input.shape.dataExtent);
        qact_shape[-1] = K / 2;
        qact.act = Tensor::allocate(qact_shape, Tensor::INT8, device, true);
        qact.ascales = Tensor::allocate({K / 64, M}, dtype, device, true);    // group_size=64
        qact.lora_act = Tensor::allocate({M, lora_rank}, Tensor::FP32, device, true);
        qact.is_unsigned = false;

        spdlog::debug("Quantize: input={}, qact.act={}, qact.ascales={}, qact.lora_act={}", input.shape.str(), qact.act.shape.str(), qact.ascales.shape.str(), qact.lora_act.shape.str());
        quantize_w4a4_act_fuse_lora(input, qact.act, qact.ascales, _lora_down, qact.lora_act, smooth);
        // quantize_w4a4_act(input, qact.act, qact.ascales);
        // auto qoutput = to_torch(qact.act);
        // Tensor::synchronizeDevice();
        // return qoutput;

        auto qweight = getTensor(wgt);
        auto _lora_up = getTensor(lora_up);
        auto _wscales = getTensor(wscales);

        const int N = _lora_up.shape[0];
    
        Tensor out;
        Tensor bias;
        GEMM_W4A4::QuantizedActivation qout;
        Tensor next_lora;
        Tensor next_smooth;
        
        auto shape = TensorShape(qact.act.shape.dataExtent);
        shape[-1] = N;
        out = Tensor::allocate(shape, dtype, device);
        
        // spdlog::info(qweight.shape.str());
        // spdlog::info(_wscales.shape.str());
        // spdlog::info(out.shape.str());
        // spdlog::info(qact.act.shape.str());
        // spdlog::info(qact.ascales.shape.str());
        // spdlog::info(qact.lora_act.shape.str());
        // spdlog::info(_lora_up.shape.str());
        // spdlog::info(_lora_down.shape.str());
        // spdlog::info(qout.act.shape.str());
        // spdlog::info(qout.ascales.shape.str());
        // spdlog::info(qout.lora_act.shape.str());
        // spdlog::info(bias.shape.str());
        // spdlog::info(next_smooth.shape.str());
        // spdlog::info("QuantizedGemm: qact.is_unsigned={}", qact.is_unsigned);
        gemm_w4a4(
            qact.act,
            qweight,
            out,
            {},
            qact.ascales,
            _wscales,
            {},
            {},
            qact.lora_act,
            _lora_up,
            {},
            {},
            {},
            {},
            {},
            bias,
            {},
            qact.is_unsigned,
            lora_scales
        );
        torch::Tensor output = to_torch(out);
        Tensor::synchronizeDevice();

        return output;
    }
};
