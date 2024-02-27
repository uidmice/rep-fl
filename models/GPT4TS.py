
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from einops import rearrange


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.patch_num += 1


        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'ln' in name or 'wpe' in name: # or 'mlp' in name:
                param.requires_grad = True
            elif 'mlp' in name and configs.mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if configs.use_gpu:
            device = torch.device('cuda:{}'.format(0))
            self.gpt2.to(device=device)

        # self.in_layer = nn.Linear(configs.patch_size, configs.d_model)

        self.predict_linear = nn.Linear(self.patch_size, configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * self.patch_num, configs.pred_len)
        

    def forward(self, x_enc,  mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]

        return None

    def forecast(self, x_enc):
        B, L, M = x_enc.shape   # L = seq_len
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        enc_out = x_enc - means
        stdev = torch.sqrt(
            torch.var(enc_out, dim=1, keepdim=True, unbiased=False) + 1e-5)
        enc_out /= stdev

        enc_out = rearrange(enc_out, 'b l m -> b m l')
        enc_out = self.padding_patch_layer(enc_out)
        enc_out = enc_out.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        enc_out = rearrange(enc_out, 'b m n p -> (b m) n p')

        enc_out = self.predict_linear(enc_out)

        outputs = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)

        outputs = outputs * stdev
        outputs = outputs + means
        
        return outputs
    
