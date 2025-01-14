import torch
import torchao

from nunchaku._C import QuantizedGEMMW4A4, QuantizedGEMM

from torchao.quantization.utils import group_quantize_tensor_symmetric
from torchao.utils import compute_max_diff

def pack_lora(weight: torch.Tensor, is_lora_down: bool) -> torch.Tensor:
    N, R = weight.shape

    assert N % 16 == 0
    assert R % 16 == 0
    assert weight.dtype.itemsize == 2

    weight = weight.reshape(N // 16, 16, R // 16, 16)
    weight = weight.permute(0, 2, 1, 3)

    if is_lora_down:
        weight = weight.transpose(-1, -2)

    # https://docs.nvidia.com/cuda/parallel-thread-execution/#mma-16816-b-f16
    assert weight.shape[2:] == (16, 16)
    weight = weight.reshape(*weight.shape[0:2], 2, 8, 2, 4, 2)
    weight = weight.permute(0, 1, 3, 5, 2, 4, 6)

    weight = weight.contiguous()
    weight = weight.view(N, R)

    return weight

if __name__ == "__main__":
    dtype = torch.bfloat16
    develop_dtype = torch.float32
    M = 512
    N = 512
    K = 512
    lora_rank = 32
    group_size = 64
    
    weight = torch.load("weight.pt")
    # weight = torch.randn((N, K), dtype=dtype, device="cuda")
    wscale_ref = torch.load("wscales.pt")
    # wscale = torch.ones((N, K // group_size), dtype=dtype, device="cuda")
    dweight = torch.load("dequantized.pt")
    qweight = torch.load("quantized.pt")
    N = weight.shape[0]
    K = weight.shape[1]
    tensor_view_shape = (N, K // group_size, group_size)
    
    wscale = (weight.to(develop_dtype).view(tensor_view_shape).abs().amax(dim=-1, keepdim=True)) / 7
    wscale[wscale == 0] = 1
    
    torch.testing.assert_close(wscale_ref.view(N,-1).to(dtype), wscale.view(N,-1).to(dtype))
    
    qweight_s8 = (weight.to(develop_dtype).view(tensor_view_shape).div(wscale.view(N, -1, 1).to(develop_dtype))).round_().clamp_(-7, 7).view(N, -1)
    # torch.testing.assert_close(qweight.to(dtype), qweight_)
    
    dweight = (qweight_s8.to(develop_dtype).view(N, -1, group_size) * wscale.to(develop_dtype).view(N, -1, 1)).view(N, -1).to(dtype)
    # torch.testing.assert_close(dequant, dweight)
    
    input = torch.randn((M, K), dtype=dtype, device="cuda")
    # weight = torch.randn((N, K), dtype=dtype, device="cuda")
    lora_down = torch.randn(K, lora_rank, dtype=dtype, device="cuda")
    lora_up = torch.randn(N, lora_rank, dtype=dtype, device="cuda")
    
    input_scales = (input.to(develop_dtype).view(M, -1, group_size).abs().amax(dim=-1, keepdim=True)) / 7
    input_s8 = (input.to(develop_dtype).view(M, -1, group_size).div(input_scales.to(develop_dtype).view(M, -1, 1))).round_().clamp_(-7, 7).view(M, -1).to(torch.int8)
    dinput = (input_s8.to(develop_dtype).view(M, -1, group_size).mul(input_scales.to(develop_dtype).view(M, -1, 1))).view(M, -1)

    output_ref = (
        (dinput.to(dtype) @ dweight.to(dtype).T).view(M, -1)
        + (input.to(develop_dtype) @ lora_down.to(develop_dtype) @ lora_up.to(develop_dtype).T).to(dtype)
    )
    
    smooth = torch.ones((K), dtype=dtype, device="cuda")
    m = QuantizedGEMMW4A4()
    qweight_s4 = torch.empty((N, K // 2), dtype=torch.int8, device="cuda")
    wscale_ = torch.empty((N, K // group_size), dtype=dtype, device="cuda")
    m.quantize_pack_w4a4_wgt(weight, qweight_s4, wscale_)
    
    lora_down_pack = pack_lora(lora_down, is_lora_down=True)
    lora_up_pack = pack_lora(lora_up, is_lora_down=False)
    out = m.gemm(input, qweight_s4.contiguous(), wscale_.to(dtype), lora_up_pack, lora_down_pack, smooth, [1.0, 1.0])
    max_diff = compute_max_diff(out, output_ref)
    print(max_diff)
    print(output_ref)
    print(out)
    