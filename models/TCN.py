
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from einops import rearrange


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, pred_len, seq_len, tcn_layers, dropout, device):
        super(Model, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len

        self.input_dim = input_dim

        self.emsize = emb_dim

        self.in_layer = weight_norm(nn.Linear(input_dim,  self.emsize))
        self.tcn = TemporalConvNet(self.emsize, tcn_layers * [self.emsize], kernel_size=3, dropout=dropout)

        self.out_layer = weight_norm(nn.Linear(self.seq_len, self.pred_len))
        self.decoder = weight_norm(nn.Linear(self.emsize, output_dim))

        self.tcn.to(device=device)

    def embed(self, x_enc):
        means = x_enc.mean((1,2), keepdim=True).detach()
        enc_out = x_enc - means
        stdev = torch.sqrt(
            torch.var(enc_out, dim=(1,2), keepdim=True, unbiased=False) + 1e-5)
        enc_out /= stdev
        enc_out = self.in_layer(enc_out)    #B L E   
        enc_out = self.tcn(enc_out)     
        return enc_out

    def forward(self, x_enc,  mask=None):
        B, L, M = x_enc.shape   # L = seq_len
        
        # Normalization from Non-stationary Transformer
        means = x_enc.mean((1,2), keepdim=True).detach()
        enc_out = x_enc - means
        stdev = torch.sqrt(
            torch.var(enc_out, dim=(1,2), keepdim=True, unbiased=False) + 1e-5)
        enc_out /= stdev

        enc_out = self.in_layer(enc_out)    #B L E

        enc_out = rearrange(enc_out, 'b l m -> b m l')
        enc_out = self.tcn(enc_out) ##B E L
        outputs = self.out_layer(enc_out.reshape(B*self.emsize, -1))  # (B*M, L) -> (B*M, pred_len)
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        outputs = self.decoder(outputs) #B L M

        outputs = outputs * stdev
        outputs = outputs + means
        
        return outputs


    

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)