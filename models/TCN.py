
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from einops import rearrange


class Model(nn.Module):
    def __init__(self, input_dim, output_dim, emb_dim, pred_len, seq_len, tcn_layers, dropout, gl=0):
        super(Model, self).__init__()
        self.pred_len = pred_len
        self.seq_len = seq_len

        self.input_dim = input_dim

        self.emsize = emb_dim

        self.in_layer = weight_norm(nn.Linear(input_dim,  self.emsize))
        self.tcn = TemporalConvNet(self.emsize, (tcn_layers - 1) * [self.emsize] + [3 * input_dim], kernel_size=3, dropout=dropout)
        self.gl = gl
        if gl > 0:
            self.gl_embed = weight_norm(nn.Linear(gl, self.seq_len))
         
        self.decoder = weight_norm(nn.Linear(3 * input_dim + int(gl>0), output_dim))

        self.out_layer = weight_norm(nn.Linear(self.seq_len, self.pred_len))


    def embed(self, x_enc):
        with torch.no_grad():
            enc_out = self.in_layer(x_enc)    #B L E   
            enc_out = rearrange(enc_out, 'b l m -> b m l')
            enc_out = self.tcn(enc_out) ##B E L
        return rearrange(enc_out, 'b m l -> b l m')

    def forward(self, x_enc):
        B, L, M = x_enc.shape   # L = seq_len

        if self.gl > 0:
            x = x_enc[:, :, :-self.gl]
        else:
            x = x_enc

        enc_out = self.in_layer(x)    #B L E

        enc_out = rearrange(enc_out, 'b l m -> b m l')
        enc_out = self.tcn(enc_out) ##B 3 L
        if self.gl > 0:
            gl_embed = self.gl_embed(x_enc[:, -1:, -self.gl:])  # B 1 L
            enc_out = torch.cat([enc_out, gl_embed], dim=1)  # B 3 L

        outputs = rearrange(enc_out, 'b m l -> b l m') #B L 3
        outputs = self.decoder(outputs) #B input_L output_dim
        outputs = rearrange(outputs, 'b l m -> b m l') #B output_dim L

        outputs = self.out_layer(outputs)  # (B output_dim L) -> (B M pred_len)
        outputs = rearrange(outputs, 'b m l -> b l m')

        # outputs = outputs * stdev
        # outputs = outputs + means
        
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