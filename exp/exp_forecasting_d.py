from data_provider.data_factory import data_provider
from utils.metrics import metric
import torch
import torch.nn as nn
import os, time, pickle
import warnings
import numpy as np
from models import TCN, ts2vec
from exp.client import Client
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from utils.tools import EarlyStopping

warnings.filterwarnings('ignore')


dataset_setting_dict = {   
    'weather1': {
        'in_partition': [0,1,2,3],
        'in_glrep': torch.LongTensor([3]),
        'out_dim': 1,
        'labels': [ 'Tdew (degC)', 
                   'rh (%)',
                   'sh (g/kg)', 
                    'rho (g/m**3)', 
                   'H2OC (mmol/mol)'],
        'dataset': 'weather'
    },
    'weather2': {
        'in_partition': [0,1,2,3],
        'in_glrep': torch.LongTensor([0,1,2]),
        'out_dim': 1,
        'labels': [ 'Tdew (degC)', 
                   'rh (%)',
                   'sh (g/kg)', 
                    'rho (g/m**3)', 
                   'H2OC (mmol/mol)'],
        'dataset': 'weather'
    },
    'weather3':{
        'dataset': 'weather',
        'in_partition': [0,2,4,5],
        'in_glrep': torch.LongTensor([5]),
        'out_dim': 1,
        'labels': [ 'rh (%)','p (mbar)', 
                   'Tdew (degC)','Tpot (K)',
                   'sh (g/kg)', 
                   'rho (g/m**3)',
                   'H2OC (mmol/mol)']
    },
    'weather4':{
        'dataset': 'weather',
        'in_partition': [0,2,4,5],
        'in_glrep': torch.LongTensor([0,1,2,3,4,5]),
        'out_dim': 1,
        'labels': [ 'rh (%)','p (mbar)', 
                   'Tdew (degC)','Tpot (K)',
                   'sh (g/kg)', 
                   'rho (g/m**3)',
                   'H2OC (mmol/mol)']
    },
    'swat': {   
        'in_partition': [0,3,7],
        'in_glrep': torch.LongTensor([5,6,7,8]),
        'out_dim': 1,
        'labels':['FIT401', 'AIT402','LIT401',
                  'FIT504', 'FIT503','FIT502', 'FIT501','PIT501','PIT503',
                  'Normal/Attack'],
        'dataset':'swat'
    }
}   



class Exp_Forecast_d:
    def __init__(self, args):
        self.args = args
        self.partition = dataset_setting_dict[args.exp_id]['in_partition']
        self.m_output = dataset_setting_dict[args.exp_id]['out_dim']
        self.args.data = dataset_setting_dict[args.exp_id]['dataset']
        self.num_clients = len(self.partition) - 1
        self.in_glm = dataset_setting_dict[args.exp_id]['in_glrep'] 
        if args.exp_id == 'swat':
            # self.args.pred_len = 1
            # self.args.label_len = args.pred_len -
            self.args.task = 'ad'
            self.args.batch_size = 256
        if not args.distributed:
            self.num_clients = 1
            self.partition = [self.partition[0], self.partition[-1]]

        if not self.args.glrep:
            self.args.glrep_dim = 0
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
            model = TCN.Model(e - s, e - s, self.args.emb_dim, 
                              self.args.pred_len, self.args.seq_len, 
                              self.args.e_layers, self.args.dropout, gl=self.args.glrep_dim).to(self.device)
            if self.args.use_multi_gpu and self.args.use_gpu:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)
            clients.append(Client(model, i, self.device, self.args))
        if self.args.glrep:
            self.fm = ts2vec.TS2Vec(len(self.in_glm), self.args.glrep_dim, self.args.att_out, 
                                self.args.g_layers, self.device, self.args)
        return clients
    

    def _get_data(self, flag, glrep=False):
        data_set, data_loader = data_provider(self.args, flag, gl=self.args.glrep_dim, labels=dataset_setting_dict[self.args.exp_id]['labels'])
        if glrep:
            local_rep = self.get_local_rep(data_set)
            gl_encode = self.fm.encode(local_rep).detach().cpu().numpy() 
            data_set.data_x[:, -self.args.glrep_dim:] = gl_encode[0]
        return data_set, data_loader
    
    def get_local_rep(self, dataset, lcrep=False):
        # return torch.cat([c.get_local_rep(dataset, self.partition[c.id], self.partition[c.id+1], lcrep) for c in self.clients], 
        #                         dim=-1)
        return torch.index_select(torch.from_numpy(dataset.data_x), 1, self.in_glm).to(torch.float).unsqueeze(0).to(self.device)

    
    def pre_train_fm(self, path):
        try:
            self.fm.net.load_state_dict(torch.load(path + '/checkpoint_ts2vec.pth', map_location=torch.device(self.device)))
        except:
            train_data, train_loader = data_provider(self.args, flag='train')
            vali_data, vali_loader = data_provider(self.args, flag='val')
            train_x = self.get_local_rep(train_data)
            val_x = self.get_local_rep(vali_data)

            early_stopping = EarlyStopping(patience=5, verbose=True)

            for _ in range(self.args.train_epochs):
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

        if self.args.glrep:
            self.pre_train_fm(path)

        train_data, train_loader = self._get_data(flag='train', glrep=self.args.glrep)
        vali_data, vali_loader = self._get_data(flag='val', glrep=self.args.glrep)

        train_steps = len(train_loader)
        records = []
        records_vali = []


        time_now = time.time()  
        for epoch in range(self.args.train_epochs):
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
                    y = batch_y[:,-self.args.pred_len:, self.partition[c.id]:self.partition[c.id+1]]
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

        test_data, test_loader = self._get_data(flag='test', glrep=self.args.glrep)

        preds = [[] for _ in range(self.partition[-1] - self.partition[0])]
        trues = [[] for _ in range(self.partition[-1] - self.partition[0])]

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y[:,-self.args.pred_len:, :]

                for c in self.clients:
                    x = batch_x[:, :, self.partition[c.id]:self.partition[c.id+1]]
                    if self.args.glrep:
                        x = torch.cat([x, batch_x[:,:,-self.args.glrep_dim:]], dim=-1)
                    outputs = c.test(x)
                    for j in range(self.partition[c.id], self.partition[c.id+1]):
                        preds[j].append(outputs[:, :, j-self.partition[c.id]:j-self.partition[c.id]+1].float().detach().cpu().numpy())
                        trues[j].append(batch_y[:, :, j-self.partition[c.id]:j-self.partition[c.id]+1].float().detach().cpu().numpy())
        f = open(os.path.join(self.args.store, f"result_{self.args.exp_id}_d.txt"), 'a')
        f.write(setting + "  \n")
        for i in range(len(preds)):
            pred = np.concatenate(preds[i], axis=0)
            true = np.concatenate(trues[i], axis=0)
            print(f'data {i} test shape:', pred.shape, true.shape)

        # if self.args.task == 'ad':
        #     preds = preds[:,0,0]
        #     trues = trues[:,0,0]
        #     accuracy = np.mean(preds==trues)
        #     precision, recall, f_score, support = precision_recall_fscore_support(trues, preds, average='binary')
        #     print("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #         accuracy, precision,
        #         recall, f_score))
        #     f.write("Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f} ".format(
        #     accuracy, precision,
        #     recall, f_score))
        # else:
            mae, mse, rmse, mape, mspe = metric(pred, true)
            print('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}'.format(mae, mse, rmse, mape, mspe))
            
            f.write('mae:{:.4f}, mse:{:.4f}, rmse:{:.4f}, mape:{:.4f}, mspe:{:.4f}\n'.format(mae, mse, rmse, mape, mspe))
        f.write('\n')
        f.close()
        return
