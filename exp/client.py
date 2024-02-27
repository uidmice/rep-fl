
import torch
from torch import optim
import torch.nn as nn
import numpy as np
from utils.tools import EarlyStopping



class Client:
    def __init__(self, model, id, device, args):
        self.id = id
        self.model = model
        self.optim = optim.Adam(model.parameters(), lr=args.learning_rate)
        self.criterion = nn.MSELoss()
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optim, T_max=10, eta_min=1e-8)
        self.early_stopping = EarlyStopping(patience=args.patience, verbose=True)
        self.stop = False
        self.args = args
        self.device = device

    
    def get_local_rep(self, dataset, s, e):
        x = torch.from_numpy(dataset.data_x[:, s:e]).to(torch.float).unsqueeze(0) # 1, T, M
        if self.args.lcrep:
            self.model.eval()
            return self.model.embed(x)
        return x



    def step(self, x, y):
        if self.stop:
            outputs = self.test(x)
            return outputs, self.criterion(outputs, y).item()
        
        self.model.train()
        self.optim.zero_grad()
        outputs = self.model(x)
        loss = self.criterion(outputs, y)
        loss.backward()
        self.optim.step()
        return outputs, loss.item()
    
    def validate(self, data_loader, path, s, e):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for batch_x, batch_y in data_loader:
                x = batch_x[:, :, s:e].float().to(self.device)
                y = batch_y[:,-self.args.pred_len:, -1].float().unsqueeze(-1).to(self.device)
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                total_loss.append(loss.item() * batch_x.size(0))

        loss = np.sum(total_loss)/len(data_loader.dataset) 
        if not self.stop:
            self.early_stopping(loss, self.model, path, self.id)
            if self.early_stopping.early_stop:
                self.stop = True
            self.scheduler.step()
            print(f'client {self.id}: validation loss {loss}, lr {self.optim.param_groups[0]["lr"]}  ')
        return loss

    def test(self, x):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x)
        return outputs
    

