import torch
import torchao

from nunchaku._C import QuantizedGEMMW4A4, QuantizedGEMM

from torchao.quantization.utils import group_quantize_tensor_symmetric
from torchao.utils import compute_max_diff

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
    
    input = torch.ones((M, K), dtype=dtype, device="cuda")
    # weight = torch.randn((N, K), dtype=dtype, device="cuda")
    lora_down = torch.zeros((K, lora_rank), dtype=dtype, device="cuda")
    lora_up = torch.zeros((N, lora_rank), dtype=dtype, device="cuda")
    
    input_scales = (input.to(develop_dtype).view(M, -1, group_size).abs().amax(dim=-1, keepdim=True)) / 7
    input_s8 = (input.to(develop_dtype).view(M, -1, group_size).div(input_scales.to(develop_dtype).view(M, -1, 1))).round_().clamp_(-7, 7).view(M, -1).to(torch.int8)
    dinput = (input_s8.to(develop_dtype).view(M, -1, group_size).mul(input_scales.to(develop_dtype).view(M, -1, 1))).view(M, -1)

    output_ref = (
        (dinput.to(dtype) @ dweight.to(dtype).T).view(M, -1)
        + (input.to(dtype) @ lora_down.to(dtype) @ lora_up.to(dtype).T)
    )
    
    smooth = torch.ones((K), dtype=dtype, device="cuda")
    m = QuantizedGEMMW4A4()
    qweight_s4 = torch.empty((N, K // 2), dtype=torch.int8, device="cuda")
    wscale_ = torch.empty((N, K // group_size), dtype=dtype, device="cuda")
    m.quantize_pack_w4a4_wgt(weight, qweight_s4, wscale_)
    
    out = m.gemm(input, qweight_s4.contiguous(), wscale_.to(dtype), lora_up, lora_down, smooth, [1.0, 1.0])
    max_diff = compute_max_diff(out, output_ref)
    print(output_ref)
    print(out)
    