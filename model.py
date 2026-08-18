import torch
import torch.nn as nn 
import torch.nn.functional as F
from tokenizer import load_tokenizer
from MLA_Transformers_MPT import Transformer,MTP
from transformers import AutoConfig,AutoTokenizer,AutoModelForCausalLM
class DeepSeek_R1(nn.Module):
    def __init__(self,cfg,model_id="deepseek-ai/DeepSeek-R1"):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = load_tokenizer(model_id)
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(num_embeddings= self.vocab_size,
                                      embedding_dim=cfg['d_in'])
        
        self.transformer_blocks = nn.Sequential(*[Transformer(d_in = cfg['d_in'],
                                                                 d_latent = cfg['d_latent'],
                                                                 d_out = cfg['d_out'],
                                                                 num_heads = cfg['num_heads'],
                                                                 RoPE_dim=cfg['RoPE_dim'],
                                                                 context_length=cfg['context_length'],
                                                                 MoE = i >= cfg['num_FFN_transformers'])
                                                        for i in range(cfg['num_transformers'])
                                                    ])
        self.output_head = nn.Linear(cfg['d_out'],self.vocab_size)
        self.MPT_Module = nn.ModuleList([
            MTP(d_in = cfg['d_in'],
                d_latent = cfg['d_latent'],
                d_out = cfg['d_out'],
                num_heads = cfg['num_heads'],
                RoPE_dim=cfg['RoPE_dim'],
                context_length=cfg['context_length'],
                vocab_size=self.vocab_size)
            for i in range(2)
        ])
        self.loss_func = nn.CrossEntropyLoss(ignore_index=-100)
    def forward(self, text_list=None, input_ids=None, attention_mask=None): # For 2 type of inputs
        if input_ids is None:
            encoded = self.tokenizer(
                text_list,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.cfg['context_length']
            )
            input_ids = encoded['input_ids'].to(next(self.parameters()).device)
            attention_mask = encoded['attention_mask'].to(next(self.parameters()).device)
        B, S = input_ids.shape
        x = self.embedding(input_ids)  # [B, S, d_in]

        for block in self.transformer_blocks:
            x = block(x, attn_mask=attention_mask)  # Hidden states: [B, S, d_out]
        logits_main = self.output_head(x) # [B,seq_len,vocab_size]

        #MPT
        mtp_logits_list = []
        current_hidden = x
        for mtp_layer in self.MPT_Module:
            # Mỗi MTP module nhận hidden states hiện tại để dự đoán token xa hơn
            logits_mtp, current_hidden = mtp_layer(self.embedding(input_ids), current_hidden)
            mtp_logits_list.append(logits_mtp)

        # If undergoing the reference process, return logits_main and mtp_logits for predict next token !
        if not self.training:
            return logits_main, mtp_logits_list
        target_ids = input_ids.clone() # [B,seq_len]
        if attention_mask is not None:
            target_ids = target_ids.masked_fill(attention_mask == 0, -100)
        shift_logits_main = logits_main[:, :-1, :].contiguous().view(-1, self.vocab_size)
        shift_targets_main = target_ids[:, 1:].contiguous().view(-1)
        loss_main = self.loss_func(shift_logits_main, shift_targets_main)
        # 4.2 Loss MTP
        total_loss = loss_main
        mtp_weight = self.cfg['mtp_loss_weight']
        for depth, logits_mtp in enumerate(mtp_logits_list, start=2):
            if S > depth:
                #Good slice :> Take Note
                shift_logits_mtp = logits_mtp[:, :-depth, :].contiguous().view(-1, self.vocab_size)
                shift_targets_mtp = target_ids[:, depth:].contiguous().view(-1)
                loss_mtp = self.loss_func(shift_logits_mtp, shift_targets_mtp)
                total_loss += mtp_weight * loss_mtp
        return total_loss
    @classmethod
    def from_pretrained(cls, model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"):
        """
        Phương thức nạp mô hình Pretrained trực tiếp từ Hugging Face vào DeepSeek_R1
        """
        print(f"1. Đang đọc Cấu hình (Config) từ HuggingFace: {model_id}...")
        hf_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        # Tự động ánh xạ tham số từ HuggingFace sang cfg của bạn
        cfg = {
            'd_in': hf_config.hidden_size,                # 1536
            'd_out': hf_config.hidden_size,               # 1536
            'd_latent': hf_config.intermediate_size,      # 8960
            'num_heads': hf_config.num_attention_heads,   # 12
            'RoPE_dim': 60,
            'context_length': 2048,
            'num_transformers': hf_config.num_hidden_layers, # 28
            'num_FFN_transformers': hf_config.num_hidden_layers, # Dùng FFN cho toàn bộ
            'num_mtp_layers': 2
        }
        # Khởi tạo Instance của DeepSeek_R1 với cfg tương ứng
        print("2. Đang khởi tạo mô hình DeepSeek_R1 theo cấu hình Pretrained...")
        model = cls(cfg = cfg,model_id = model_id)
        print("3. Đang nạp trọng số (Weights) từ Hugging Face...")
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        hf_sd = hf_model.state_dict()
        custom_sd = model.state_dict()
        # Ánh xạ tên Layer từ Hugging Face sang Custom Model
        new_sd = {}
        
        # 1. Embedding & Output Head
        if "model.embed_tokens.weight" in hf_sd:
            new_sd["embedding.weight"] = hf_sd["model.embed_tokens.weight"]
        if "lm_head.weight" in hf_sd:
            new_sd["output_head.weight"] = hf_sd["lm_head.weight"]
        # 2. Ánh xạ các tầng Transformer Blocks
        for i in range(cfg['num_transformers']):
            prefix_hf = f"model.layers.{i}."
            prefix_custom = f"transformer_blocks.{i}."
            # RMSNorm 1 & 2
            if f"{prefix_hf}input_layernorm.weight" in hf_sd:
                new_sd[f"{prefix_custom}rmsnorm1.gamma"] = hf_sd[f"{prefix_hf}input_layernorm.weight"]
            if f"{prefix_hf}post_attention_layernorm.weight" in hf_sd:
                new_sd[f"{prefix_custom}rmsnorm2.gamma"] = hf_sd[f"{prefix_hf}post_attention_layernorm.weight"]
            # Output Projection Attention
            if f"{prefix_hf}self_attn.o_proj.weight" in hf_sd:
                if f"{prefix_custom}attenion_head.out_proj.weight" in custom_sd:
                    if custom_sd[f"{prefix_custom}attenion_head.out_proj.weight"].shape == hf_sd[f"{prefix_hf}self_attn.o_proj.weight"].shape:
                        new_sd[f"{prefix_custom}attenion_head.out_proj.weight"] = hf_sd[f"{prefix_hf}self_attn.o_proj.weight"]
        # Nạp trọng số vào mô hình với strict=False
        missing, unexpected = model.load_state_dict(new_sd, strict=False)
        print(f"   --> Nạp thành công {len(new_sd)} ma trận trọng số!")
        print(f"   --> Missing Keys (Các layer chưa có trọng số gốc như MTP): {len(missing)}")
        return model

        

if __name__ == "__main__":
    # Nạp mô hình Pretrained 1.5B trực tiếp qua model.py
    model = DeepSeek_R1.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    print("\nTrạng thái: Mô hình DeepSeek_R1 của bạn đã sẵn sàng chạy!")


        

    
        
    