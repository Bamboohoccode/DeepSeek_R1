import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import RoPE,RMSNorm,FFN,DeepSeekMoE


class Multi_Latent_Attention(nn.Module):
    def __init__(self,d_in,d_latent,d_out,num_heads,RoPE_dim,context_length):
        super().__init__()
        assert (d_out + RoPE_dim) % num_heads == 0
        assert RoPE_dim % num_heads == 0
        self.d_latent = d_latent
        self.d_out = d_out
        self.num_heads = num_heads
        self.dim_head = (d_out + RoPE_dim) // self.num_heads

        self.W_cq = nn.Linear(d_in,d_latent)
        self.W_ckv = nn.Linear(d_in,d_latent)
        self.rope = RoPE(RoPE_dim // num_heads)

        self.W_uq = nn.Linear(d_latent,d_out)
        self.W_qr = nn.Linear(d_latent,RoPE_dim)

        self.W_kr = nn.Linear(d_in,RoPE_dim)
        self.W_uk = nn.Linear(d_latent,d_out)
        self.W_uv = nn.Linear(d_latent,d_out)
        self.out_proj = nn.Linear(self.dim_head - RoPE_dim // num_heads,d_out)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self,X):
        b,seq_len,d = X.shape

        c_tq = self.W_cq(X)
        c_tkv = self.W_ckv(X)
        # Query
        q_c = self.W_uq(c_tq) #[b,seq_len,d_out]
        reshaped_q_c = q_c.view(b,seq_len,self.num_heads,-1)
        q_r = self.W_qr(c_tq)
        reshaped_q_r = q_r.view(b,seq_len,self.num_heads,-1)
        reshaped_q_r = self.rope(reshaped_q_r)

        concat_q = torch.concat([reshaped_q_c,reshaped_q_r],dim = -1) # [b,seq_len,d_out + RoPE_dim]
        # Key
        k_c = self.W_uk(c_tkv)
        reshaped_k_c = k_c.view(b,seq_len,self.num_heads,-1)
        k_r = self.W_kr(X) # [b,seq_len,RoPE_dim]
        reshaped_k_r = k_r.view(b,seq_len,self.num_heads,-1)
        reshaped_k_r = self.rope(reshaped_k_r)

        concat_k = torch.concat([reshaped_k_c,reshaped_k_r],dim = -1)  # [b,seq_len,d_out + RoPE_dim]

        v_c = self.W_uv(c_tkv)


        Q = concat_q.view(b,seq_len,self.num_heads,-1).transpose(1,2).contiguous() # [b,num_heads,seq_len,dim_head]
        K = concat_k.view(b,seq_len,self.num_heads,-1).transpose(1,2).contiguous() # [b,num_heads,seq_len,dim_head]
        V = v_c.view(b,seq_len,self.num_heads,-1).transpose(1,2).contiguous() # [b,num_heads,seq_len,dim_head - RoPE_dim // num_heads]

        attn_scores = Q @ K.transpose(2,3)
        mask_bool = self.mask.bool()[:seq_len,:seq_len]
        attn_scores = attn_scores.masked_fill(mask_bool,-torch.inf)

        attn_scores = torch.softmax(attn_scores / (self.dim_head ** 0.5),dim = -1)
        context_vec = attn_scores @ V # [b,num_heads,seq_len,dim_head - RoPE_dim // num_heads]
        context_vec = context_vec.transpose(1,2).contiguous().view(b,seq_len,-1)
        context_vec = self.out_proj(context_vec)
        return context_vec # [b,seq_len,d_out]


class Transformer(nn.Module):
    def __init__(self,d_in,d_latent,d_out,num_heads,RoPE_dim,context_length,MoE : bool = False):
        self.rmsnorm1 = RMSNorm(d_in)
        self.attenion_head = Multi_Latent_Attention(d_in = d_in,
                                                    d_latent=d_latent,
                                                    d_out = d_out,
                                                    num_heads=num_heads,
                                                    RoPE_dim=RoPE_dim,
                                                    context_length=context_length)
        self.rmsnorm2 = RMSNorm(d_out)

        if MoE:
            self.out_proj = DeepSeekMoE(d_out= d_out,
                                        d_ffn_routed=d_latent,
                                        d_ffn_shared=d_latent)
        else:
            self.out_proj = FFN(d_out = d_out,
                                d_latent=d_latent)
    def forward(self,X):
        Branch_X1 = self.rmsnorm1(X)
        Branch_X1 = self.attenion_head(Branch_X1)
        X = X + Branch_X1
        Branch_X2 = self.rmsnorm2(X)
        Branch_X2 = self.out_proj(X)
        X = X + Branch_X2
        return X
    
class MTP(nn.Module):
    def __init__(self,d_in,d_latent,d_out,num_heads,RoPE_dim,context_length,vocab_size):
        self.rmsnorm1 = RMSNorm(emb_dim=d_in)
        in_features = d_in + d_out
        self.linear_proj = nn.Linear(in_features=in_features,
                                     out_features = d_out)
        self.transformer = Transformer(d_in = d_in,
                                    d_latent = d_latent,
                                    num_heads = num_heads,
                                    RoPE_dim=RoPE_dim,
                                    context_length=context_length,
                                    MoE= False)
        self.output_proj = nn.Linear(d_out,vocab_size)
        self.rmsnorm2 = RMSNorm(emb_dim=d_out)
    def forward(self,X,Add_layer):
        Add_layer = self.rmsnorm2(Add_layer)
        concat_X = torch.concat([X,Add_layer],dim = -1) # [4,seq_len,d_in + d_out]
        output = self.transformer(self.linear_proj(concat_X))

        return self.output_proj(output),output
    



         

