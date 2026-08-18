import torch
import torch.nn as nn
import torch.nn.functional as F
class RoPE(nn.Module):
    def __init__(self,d,max_seq_len = 2048,base : int = 10000):
        super().__init__()
        # Co the khong can luu
        self.base = base
        theta = 1.0 / (self.base ** (torch.arange(0,d,2).float() / d))
        
        idx = torch.arange(max_seq_len).float()
        
        idx_theta = torch.einsum('x,y->xy',idx,theta) # Tao ra ma tran (seq_len,d//2)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=-1)
        
        self.register_buffer('cos_theta' , idx_theta2.cos()[None,:,None,:])
        self.register_buffer('sin_theta' , idx_theta2.sin()[None,:,None,:])
        
    def neg_half(self,X):
        d_2 = X.shape[-1] // 2
        return torch.cat([-X[...,d_2:],X[...,:d_2]],dim = -1)
    def forward(self,X):
        neg_half = self.neg_half(X)
        seq_len = X.shape[1]
        if X.ndim == 3:
            cos = self.cos_theta[:,:seq_len,0,:]
            sin = self.sin_theta[:,:seq_len,0,:]
        else:
            cos = self.cos_theta[:,:seq_len,:,:]
            sin = self.sin_theta[:,:seq_len,:,:]
        return (X * cos) + (neg_half * sin )
    print("Success")

class  RMSNorm(nn.Module):
    def __init__(self,emb_dim,e = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(emb_dim))
        self.epsilon = e
    def forward(self,X):
        variance = (X.pow(2)).mean(dim = -1,keepdim = True)
        X_normed = X * torch.rsqrt(variance + self.epsilon)
        return self.gamma * X_normed

class FFN(nn.Module):
    def __init__(self,d_out,d_latent):
        super().__init__()
        self.W1 = nn.Linear(d_out,d_latent)
        self.activation = nn.SiLU()
        self.W2 = nn.Linear(d_latent,d_out)
    def forward(self,X):
        X = self.W1(X)
        X = self.activation(X)
        output = self.W2(X)
        return output

class DeepSeekMoE(nn.Module):
    """
    Mô hình DeepSeekMoE kết hợp:
    1. Shared Experts (Chuyên gia dùng chung - Luôn bật cho mọi token)
    2. Routed Experts (Chuyên gia định tuyến - Chỉ chọn Top-K chuyên gia cho mỗi token)
    """
    def __init__(
        self,
        d_out: int = 4096,         # Kích thước Hidden Dimension của Transformer
        d_ffn_routed: int = 2048,    # Kích thước ẩn của 1 Routed Expert (Thường chia nhỏ)
        d_ffn_shared: int = 2048,    # Kích thước ẩn của Shared Expert
        num_routed_experts: int = 64,# Tổng số Routed Experts
        num_shared_experts: int = 1, # Số lượng Shared Experts
        top_k: int = 6               # Số Routed Experts được chọn cho 1 token
    ):
        super().__init__()
        self.d_out = d_out
        self.num_routed = num_routed_experts
        self.top_k = top_k
        self.router = nn.Linear(d_out, num_routed_experts, bias=False)

        total_shared_dim = num_shared_experts * d_ffn_shared
        self.shared_experts = FFN(d_out, total_shared_dim)

        self.routed_experts = nn.ModuleList([
            FFN(d_out, d_ffn_routed) for _ in range(num_routed_experts)
        ])

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Input X Shape : [Batch_Size, Seq_Len, d_out]
        Output Shape  : [Batch_Size, Seq_Len, d_out]
        """
        B, S, d_out = X.shape
        N_total = B * S
        
        # Duỗi phẳng 3D [B, S, d_out] thành 2D [N_total, d_out] để tính toán
        X_flat = X.view(N_total, d_out)

        out_shared = self.shared_experts(X_flat)

        router_logits = self.router(X_flat)

        # topk_weights: [N_total, top_k], topk_indices: [N_total, top_k]
        topk_weights, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)

        topk_weights = F.softmax(topk_weights, dim=-1)

        out_routed = torch.zeros_like(X_flat)  # Tensor rỗng lưu kết quả: [N_total, d_out]

        for k in range(self.top_k):
            expert_indices = topk_indices[:, k]       # Indices của expert được chọn ở vị trí k: [N_total]
            routing_weights = topk_weights[:, k:k+1]

            for expert_id in range(self.num_routed):
                token_mask = (expert_indices == expert_id)

                if token_mask.any():
                    selected_tokens = X_flat[token_mask]
                    
                    expert_output = self.routed_experts[expert_id](selected_tokens)
                    
                    out_routed[token_mask] += expert_output * routing_weights[token_mask]

        out_final = out_shared + out_routed

        return out_final.view(B, S, d_out)


