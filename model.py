import torch
import torch.nn as nn 
import torch.nn.functional as F
from tokenizer import load_tokenizer
from MLA_Transformers_MPT import Transformer,MTP

class DeepSeek_R1(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = load_tokenizer()
        self.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(num_embeddings= self.vocab_size,
                                      embedding_dim=cfg['d_in'])
        
        self.transformer_blocks = nn.Sequential(*[Transformer(d_in = cfg['d_in'],
                                                                 d_latent = cfg['d_latent'],
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
        




        

    
        
    