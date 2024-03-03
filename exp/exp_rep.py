from data_provider.data_factory import data_provider
from utils.metrics import metric
import torch
import torch.nn as nn
import os, time, pickle
import warnings
import numpy as np
from models import TCN, ts2vec
from exp.client import Client
from einops import rearrange
from utils.tools import EarlyStopping

warnings.filterwarnings('ignore')


dataset_setting_dict = {   
    'weather': {
        'in_partition': [0,1,2,3,4,5],
        'out_dim': 1,
        'labels': [ 'Tdew (degC)', 'T (degC)', 'Tpot (K)', 
                'VPmax (mbar)', 'rho (g/m**3)', 'VPdef (mbar)','Tlog (degC)'],
        'dataset': 'weather'
    },
    'weather2':{
        'dataset': 'weather',
        'in_partition': [0,1,2,3,4,5],
        'out_dim': 1,
        'labels': [ 'PAR (�mol/m�/s)', 'Tpot (K)', 
                 'VPmax (mbar)', 'rho (g/m**3)', 'VPdef (mbar)','Tlog (degC)']
    }
}



class Exp_Rep:
    def __init__(self, args):
        self.args = args
        self.partition = dataset_setting_dict[args.exp_id]['in_partition']
        self.m_output = dataset_setting_dict[args.exp_id]['out_dim']
        self.args.data = dataset_setting_dict[args.exp_id]['dataset']
        self.num_clients = len(self.partition) - 1

        if not args.distributed:
            self.num_clients = 1
            self.partition = [self.partition[0], self.partition[-1]]
            
        self.device = self._acquire_device()
        self.clients = self._build_model()
    
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device
    
    def _build_model(self):
        clients = []
        for i in range(self.num_clients):
            s = self.partition[i]
            e = self.partition[i+1]
            model = TCN.Model(e - s, self.m_output, self.args.emb_dim, 
                              self.args.pred_len, self.args.seq_len, 
                              self.args.e_layers, self.args.dropout, gl=self.args.glrep_dim).to(self.device)
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
            clients.append(Client(model, i, self.device, self.args))
        self.fm = ts2vec.TS2Vec((self.partition[-1] - self.partition[0]) * 3, self.args.glrep_dim, self.args.att_out, 
                                self.args.g_layers, self.device, self.args)
        return clients
    

    def _get_data(self, flag, glrep=False):
        data_set, data_loader = data_provider(self.args, flag, gl=self.args.glrep_dim, labels=dataset_setting_dict[self.args.exp_id]['labels'])
        if glrep:
            local_rep = self.get_local_rep(data_set, True)
            gl_encode = self.fm.encode(local_rep).detach().cpu().numpy() 
            data_set.data_x[:, -self.args.glrep_dim:] = gl_encode[0]
        return data_set, data_loader
    
    def get_local_rep(self, dataset, lcrep=False):
        return torch.cat([c.get_local_rep(dataset, self.partition[c.id], self.partition[c.id+1], lcrep) for c in self.clients], 
                                dim=-1)
    
    def train_fm(self, path, iters):
        train_data, train_loader = data_provider(self.args, flag='train')
        vali_data, vali_loader = data_provider(self.args, flag='val')
        train_x = self.get_local_rep(train_data, True)
        val_x = self.get_local_rep(vali_data, True)
        
        early_stopping = EarlyStopping(patience=5, verbose=True)

        for _ in range(iters):
            train_loss = self.fm.fit(train_x, verbose=True)
            print('train loss:', np.average(train_loss))
            vali_loss = self.fm.fit(val_x, verbose=True, validate=True)
            early_stopping(vali_loss[0], self.fm.net, path, 'ts2vec')

            if early_stopping.early_stop:
                self.fm.net.load_state_dict(torch.load(path + '/checkpoint_ts2vec.pth'))
                break


    def train(self, setting):
        path_r = os.path.join(self.args.store, setting)
        if not os.path.exists(path_r):
            os.makedirs(path_r)

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)


        records = []
        records_vali = []


        time_now = time.time()  
        next_fm_train = 1
        last_fm_train = -1
        for epoch in range(self.args.train_epochs):
            train_data, train_loader = self._get_data(flag='train', glrep=(epoch>0))
            vali_data, vali_loader = self._get_data(flag='val', glrep=(epoch>0))

            train_steps = len(train_loader)

            iter_count = 0
            train_loss = []
            ep_loss = [[] for _ in range(self.num_clients)]


            epoch_time = time.time()
            for i, (batch_x, batch_y) in enumerate(train_loader):
                iter_count += 1

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                err = []
                for c in self.clients:
                    x = batch_x[:, :, self.partition[c.id]:self.partition[c.id+1]]
                    if self.args.glrep:
                        x = torch.cat([x, batch_x[:,:,-train_data.gl:]], dim=-1)
                    y = batch_y[:,-self.args.pred_len:, -1].unsqueeze(-1)
                    outputs, loss = c.step(x, y)
                    err.append(loss)
                    ep_loss[c.id].append(loss)
                err = np.mean(err)
                train_loss.append(err * batch_x.size(0))

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, err))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            records.append(ep_loss)
            vali_loss = self.vali(vali_loader, path)
            records_vali.append(vali_loss)
            vali_loss = np.mean(vali_loss)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, np.sum(train_loss)/len(train_data), vali_loss))
            
            if all([c.stop for c in self.clients]):
                break

            if epoch >= last_fm_train + next_fm_train:
                self.train_fm(path, max(3, 10 - epoch))
                last_fm_train = epoch
                next_fm_train *= 2


        pickle.dump(records, open(path_r + '/train.pkl', 'wb'))
        pickle.dump(records_vali, open(path_r + '/vali.pkl', 'wb'))
        self.load_model(path)


    def load_model(self, path):
        for c in self.clients:
            c.model.load_state_dict(torch.load(path + '/checkpoint_{}.pth'.format(c.id), map_location=self.device))
        if self.args.glrep:
            self.fm.net.load_state_dict(torch.load(path + '/checkpoint_ts2vec.pth'))

    def vali(self, vali_loader, path):
        loss = []
        for c in self.clients:
            loss.append(c.validate(vali_loader, path, self.partition[c.id], self.partition[c.id+1], gl=self.args.glrep_dim))
        return loss

    def test(self, setting):
        path = os.path.join(self.args.checkpoints, setting)
        self.load_model(path)

        test_data, test_loader = self._get_data(flag='test')

        preds = []
        trues = []

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:,-self.args.pred_len:, -1].float().unsqueeze(-1).detach().cpu().numpy()
                pred = []

                for c in self.clients:
                    x = batch_x[:, :, self.partition[c.id]:self.partition[c.id+1]]
                    if self.args.glrep:
                        x = torch.cat([x, batch_x[:,:,-self.args.glrep_dim:]], dim=-1)
                    outputs = c.test(x)
                    pred.append(outputs)

                pred = torch.mean(torch.stack(pred), dim=0).detach().cpu().numpy()
                preds.append(pred)
                trues.append(batch_y)
            
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        # # result save
        # folder_path = './ett_results/' + self.args.model + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)
        print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(mae, mse, rmse, mape, mspe))
        f = open(os.path.join(self.args.store, f"result_{self.args.exp_id}.txt"), 'a')
        f.write(setting + "  \n")
        f.write('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(mae, mse, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()
        return
