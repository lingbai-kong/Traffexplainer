import torch
import torch.nn as nn
from logging import getLogger
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.data import get_dataset
from libcity.utils import get_model
from libcity.config import ConfigParser
from copy import deepcopy
import pickle
import os
torch.autograd.set_detect_anomaly(True)
class Explainer(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        config = config['Mconfig']
        super().__init__(config, data_feature)
        # section 1: data_feature
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('featrue_dim', 1)
        self.logger = getLogger()
        self.logger.info("Torch version " + torch.__version__)
        # section 2: model config
        self.input_window = config.get('input_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        self.base_model = config.get('base_model','None')
        self.is_train = config.get('train','True')
        self.exp_id=config['exp_id']
        # section 3: model structure
        self.adj_mx = torch.from_numpy(data_feature['adj_mx']).to(self.device)  # tensor(ndarray)
        self.mask = torch.zeros_like(self.adj_mx, dtype=torch.float).to(self.device)
        self.mask = torch.nn.Parameter(self.mask).to(self.device)
        self.register_parameter("Mask",self.mask)
        # self.mask_grad = torch.zeros_like(self.mask, dtype=torch.float).to(self.device)

        self.mask2 = torch.zeros_like(self.adj_mx, dtype=torch.float).to(self.device)
        self.mask2 = torch.nn.Parameter(self.mask2).to(self.device)
        self.register_parameter("Mask2",self.mask2)
        # self.mask2_grad = torch.zeros_like(self.mask2, dtype=torch.float).to(self.device)
        
        self.diag_mask = torch.ones_like(self.mask).to(self.device)-torch.diag(torch.ones(self.mask.shape[0])).to(self.device)
        
        self.feature_mask = torch.zeros((self.input_window * self.num_nodes, 1), dtype=torch.float, requires_grad=True).to(self.device)
        self.feature_mask = torch.nn.Parameter(self.feature_mask).to(self.device)
        self.register_parameter("FeatureMask",self.feature_mask)
        # self.fmask_grad = torch.zeros_like(self.feature_mask, dtype=torch.float).to(self.device)

        self.Model = get_model(config, data_feature)
        self.Model.eval()
        for param in self.Model.parameters():
            param.requires_grad = False
        
        dataset_name=config['dataset']
        if dataset_name=='TGESHD':
            # TGESHD
            cache_name_dict = {
                'mySTGCN':'./libcity/cache/96118/model_cache/mySTGCN_TGESHD.m',
                'HGCN':'./libcity/cache/44711/model_cache/HGCN_METR_LA.m',
                'TGCN':'./libcity/cache/81856/model_cache/TGCN_TGESHD.m',
                'DCRNN':'./libcity/cache/47320/model_cache/DCRNN_TGESHD.m',
                'MTGNN':'./libcity/cache/78181/model_cache/MTGNN_TGESHD.m',
            }
        elif dataset_name=='PEMSD8':
            # PEMSD8
            cache_name_dict = {
                'mySTGCN':'./libcity/cache/68375/model_cache/mySTGCN_PEMSD8.m',
                'HGCN':'./libcity/cache/17249/model_cache/HGCN_PEMSD8.m',
                'TGCN':'./libcity/cache/88817/model_cache/TGCN_PEMSD8.m',
                'DCRNN':'./libcity/cache/26360/model_cache/DCRNN_PEMSD8.m',
                'MTGNN':'./libcity/cache/74146/model_cache/MTGNN_PEMSD8.m',
                'GTS':'./libcity/cache/66189/model_cache/GTS_PEMSD8.m'
            }
        elif dataset_name=='PEMSD7':
            # PEMSD7
            cache_name_dict = {
                'mySTGCN':'./libcity/cache/5768/model_cache/mySTGCN_PEMSD7.m',
                'HGCN':'./libcity/cache/14483/model_cache/HGCN_PEMSD7.m',
                'TGCN':'./libcity/cache/90568/model_cache/TGCN_PEMSD7.m',
                'DCRNN':'./libcity/cache/75749/model_cache/DCRNN_PEMSD7.m',
                'MTGNN':'./libcity/cache/71639/model_cache/MTGNN_PEMSD7.m',
                'GTS':'./libcity/cache/35515/model_cache/GTS_PEMSD7.m'
            }
        elif dataset_name=='PEMSD4':
            # PEMSD4
            cache_name_dict = {
                'mySTGCN':'./libcity/cache/83111/model_cache/mySTGCN_PEMSD4.m',
                'HGCN':'./libcity/cache/61463/model_cache/HGCN_PEMSD4.m',
                'TGCN':'./libcity/cache/32720/model_cache/TGCN_PEMSD4.m',
                'DCRNN':'./libcity/cache/31740/model_cache/DCRNN_PEMSD4.m',
                'MTGNN':'./libcity/cache/62019/model_cache/MTGNN_PEMSD4.m',
                'GTS':'./libcity/cache/55331/model_cache/GTS_PEMSD4.m',
                'STMGAT':'./libcity/cache/82702/model_cache/STMGAT_PEMSD4.m',
                'STGODE':'./libcity/cache/44100/model_cache/STGODE_PEMSD4.m'
            }
        elif dataset_name=='PEMSD3':
            # PEMSD3
            cache_name_dict = {
                'mySTGCN':'./libcity/cache/99125/model_cache/mySTGCN_PEMSD3.m',
                'HGCN':'./libcity/cache/16176/model_cache/HGCN_PEMSD3.m',
                'TGCN':'./libcity/cache/48432/model_cache/TGCN_PEMSD3.m',
                'DCRNN':'./libcity/cache/68678/model_cache/DCRNN_PEMSD3.m',
                'MTGNN':'./libcity/cache/96014/model_cache/MTGNN_PEMSD3.m',
                'GTS':'./libcity/cache/31704/model_cache/GTS_PEMSD3.m'
            }

        if self.is_train:
            cache_name = cache_name_dict[config['base_model']]
            self.logger.info("Loaded model at " + cache_name)
            model_state, optimizer_state = torch.load(cache_name)
            self.Model.load_state_dict(model_state)
        
        if config['base_model'] == 'MTGNN':
            self.predict=self.predict_mtgnn
            self.calculate_loss=self.calculate_loss_mtgnn
        elif config['base_model'] == 'DCRNN' or config['base_model'] == 'GTS':
            self.predict=self.predict_batches_seen
            self.calculate_loss=self.calculate_loss_batches_seen
        else:
            self.predict=self.predict_default
            self.calculate_loss=self.calculate_loss_default
    
    def activated_mask(self, mask):
        mask = torch.tanh(mask)+1
        mask = (mask + mask.T)/2
        if not self.is_train:
            zero = torch.zeros_like(mask)
            one = torch.ones_like(mask)
            mask = torch.where(mask < 1, zero, mask)
            mask = torch.where(mask > 1, one, mask)
        return mask
    
    def activate_fmask(self, fmask, xdim):
        fmask = fmask.repeat(1, self.feature_dim)
        fmask = fmask.repeat(xdim, 1)
        fmask = fmask.reshape(xdim, self.input_window, self.num_nodes, self.feature_dim)
        fmask = torch.tanh(fmask)+1
        if not self.is_train:
            zero = torch.zeros_like(fmask)
            one = torch.ones_like(fmask)
            fmask = torch.where(fmask < 1, zero, fmask)
            fmask = torch.where(fmask > 1, one, fmask)
        return fmask

    def predict_default(self, batch):
        batch['X'] *= self.activate_fmask(self.feature_mask, len(batch['X']))
        self.Model.updateAdj(self.adj_mx * self.activated_mask(self.mask) * self.diag_mask,
                             self.adj_mx * self.activated_mask(self.mask2) * self.diag_mask)
        
        return self.Model.predict(batch)
    def calculate_loss_default(self, batch):
        batch['X'] *= self.activate_fmask(self.feature_mask, len(batch['X']))
        self.Model.updateAdj(self.adj_mx * self.activated_mask(self.mask) * self.diag_mask,
                             self.adj_mx * self.activated_mask(self.mask2) * self.diag_mask)
        
        loss = self.Model.calculate_loss(batch)

        # if self.mask.grad!=None and self.mask2.grad!=None:
        #     self.mask_grad += self.mask.grad
        #     self.mask2_grad += self.mask2.grad
        
        # if self.feature_mask.grad!=None:
        #     self.fmask_grad += self.feature_mask.grad
        
        return loss
    
    def predict_mtgnn(self, batch, idx=None):
        batch['X'] *= self.activate_fmask(self.feature_mask, len(batch['X']))
        
        self.Model.updateAdj(self.adj_mx * self.activated_mask(self.mask) * self.diag_mask,
                             self.adj_mx * self.activated_mask(self.mask2) * self.diag_mask)
        
        return self.Model.predict(batch, idx)
    
    def calculate_loss_mtgnn(self, batch, idx=None, batches_seen=None):
        batch['X'] *= self.activate_fmask(self.feature_mask, len(batch['X']))
        
        self.Model.updateAdj(self.adj_mx * self.activated_mask(self.mask) * self.diag_mask,
                             self.adj_mx * self.activated_mask(self.mask2) * self.diag_mask)
        
        loss = self.Model.calculate_loss(batch, idx, batches_seen)
        return loss
    
    def predict_batches_seen(self, batch, batches_seen=None):
        batch['X'] *= self.activate_fmask(self.feature_mask, len(batch['X']))
        
        self.Model.updateAdj(self.adj_mx * self.activated_mask(self.mask) * self.diag_mask,
                             self.adj_mx * self.activated_mask(self.mask2) * self.diag_mask)
        
        return self.Model.predict(batch, batches_seen)
    
    def calculate_loss_batches_seen(self, batch, batches_seen=None):
        batch['X'] *= self.activate_fmask(self.feature_mask, len(batch['X']))
        
        self.Model.updateAdj(self.adj_mx * self.activated_mask(self.mask) * self.diag_mask,
                             self.adj_mx * self.activated_mask(self.mask2) * self.diag_mask)
        
        loss = self.Model.calculate_loss(batch, batches_seen)
        return loss

    def __del__(self):
        if not self.is_train:
            return

        path = f'../../data/mask/{self.base_model}_{self.exp_id}'
        if not os.path.exists(path):
            os.makedirs(path)
        
        with open(path+'/adj.pkl','wb') as f:
            pickle.dump(self.adj_mx.detach().cpu().numpy(), f)

        mask = self.activated_mask(self.mask)
        with open(path+'/mask.pkl','wb') as f:
            pickle.dump(mask.detach().cpu().numpy(), f)
            
        # mask_grad = self.mask_grad.detach().cpu().numpy()
        # with open(path+'/mask_grad.pkl','wb') as f:
        #     pickle.dump(mask_grad, f)
            
        mask2 = self.activated_mask(self.mask2)
        with open(path+'/mask2.pkl','wb') as f:
            pickle.dump(mask2.detach().cpu().numpy(), f)
            
        # mask2_grad = self.mask2_grad.detach().cpu().numpy()
        # with open(path+'/mask2_grad.pkl','wb') as f:
        #     pickle.dump(mask2_grad, f)
            
        fmask = (torch.tanh(self.feature_mask)+1).reshape(self.input_window,self.num_nodes)
        with open(path+'/fmask.pkl','wb') as f:
            pickle.dump(fmask.detach().cpu().numpy(), f)
        
        # fmask_grad = self.fmask_grad.detach().cpu().numpy().reshape(self.input_window,self.num_nodes)
        # with open(path+'/fmask_grad.pkl','wb') as f:
        #     pickle.dump(fmask_grad, f)

        self.logger.info("Saved mask at " + path)
        