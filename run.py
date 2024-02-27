import argparse
import os
import torch


from exp.exp_forecasting_d import Exp_Forecast_D
from exp.exp_forecasting import Exp_Forecast
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Rep-FL')

# basic config

parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--exp_id', type=str, default='weather', help='model id')
parser.add_argument('--model', type=str, default='TCN',
                    help='model name, options: [TCN]')
parser.add_argument('--fm', type=str, default='TS2Vec',
                    help='model name, options: [TS2Vec]')
parser.add_argument('--distributed', type=int, help='distributed setting', default=0)
parser.add_argument('--glrep', action='store_true', help='use global fm', default=False)
parser.add_argument('--lcrep', action='store_true', help='use local representation learning', default=False)

# data loader
parser.add_argument('--data', type=str, default='weather', help='dataset type')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--store', type=str, default='./runs/', help='location of experiment results')

# forecasting task
parser.add_argument('--seq_len', type=int, default=128, help='input sequence length')
parser.add_argument('--label_len', type=int, default=10, help='start token length')
parser.add_argument('--pred_len', type=int, default=48, help='prediction sequence length')

# model define
parser.add_argument('--emb_dim', type=int, default=32, help='decoder input size')
parser.add_argument('--e_layers', type=int, default=3, help='num of tcn layers')
#fm
parser.add_argument('--att_out', type=int, default=64, help='attention output dim')
parser.add_argument('--glrep_dim', type=int, default=256, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
parser.add_argument('--g_layers', type=int, default=1, help='num of encoder layers')
# parser.add_argument('--c_out', type=int, default=7, help='output size')
# parser.add_argument('--d_model', type=int, default=768, help='dimension of model')
# parser.add_argument('--n_heads', type=int, default=4, help='num of heads')
# parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
# parser.add_argument('--d_ff', type=int, default=768, help='dimension of fcn')
# parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')

parser.add_argument('--dropout', type=float, default=0.3, help='dropout')

# optimization
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers') ########
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=30, help='train epochs')
parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')


# # patching
# parser.add_argument('--patch_size', type=int, default=16)
# parser.add_argument('--stride', type=int, default=16)
# parser.add_argument('--gpt_layers', type=int, default=6)
# parser.add_argument('--ln', type=int, default=0)
# parser.add_argument('--mlp', type=int, default=0)


args = parser.parse_args()
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)


if not args.glrep:
    Exp = Exp_Forecast_D
else:
    Exp = Exp_Forecast  
    
dist = 'dist' if args.distributed else 'no_dist'

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_sl{}_ll{}_pl{}_el{}_{}_{}'.format(
            args.exp_id,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.e_layers, dist, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting)
        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_sl{}_ll{}_pl{}_el{}_{}_{}'.format(
            args.exp_id,
            args.model,
            args.data,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.e_layers, dist, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)
    torch.cuda.empty_cache()
