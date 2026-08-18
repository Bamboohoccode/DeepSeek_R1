import torch
import torch.nn as nn 
import torch.nn.functional as F
from tokenizer import load_tokenizer
from MLA_Transformers_MPT import Transformer,MTP

class DeepSeek_R1(nn.Module):
    def __init__(self,cfg):
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
        self.loss =nn.CrossEntropyLoss()

    def forward(self,X):
        #X: List of string []
        encoded_token = self.tokenizer(X,
                                    return_tensors = 'pt',
                                    padding = True,
                                    truncation = True,
                                    max_length = self.cfg['max_length']
                                    )
        input_ids = encoded_token['input_ids']
        attn_mask = encoded_token['attention_mask']
        inputs = self.embedding(input_ids) # [b,seq_len,d_in]

        b,seq_len,d_in = inputs.shape

        inputs = inputs.view(-1,6,seq_len,d_in)
        batch_input_ids = input_ids.view(-1,6,seq_len,1)

        outputs = self.transformer_blocks(inputs[:,0:4,:,:])

        target_tokens = [self.output_head(outputs)] # Cac phan tu co shape [b,4,seq_len,vocab_size]

        for i in range(len(self.MPT_Module)):
            target_token,outputs = self.MPT_Module[i](inputs[:,i:i+4,:,:],outputs)
            target_tokens.append(target_token)
        # Sau loop ta co 3 phan tu trong target_tokens [3,b,4,seq_len,vocab_size]

        Loss = 0
        for i in range(len(target_tokens)):
            output = target_tokens[i]
            target = batch_input_ids[:,i:i+4,:,:]
            Loss += self.loss(output,target)
        return Loss
        




        

    
        
    