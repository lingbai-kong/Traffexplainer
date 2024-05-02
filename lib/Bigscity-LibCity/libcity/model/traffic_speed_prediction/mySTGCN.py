import math
import numpy as np
from logging import getLogger
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from libcity.model import loss
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

def calulate_max_eigval(matrix, device):
    x = torch.rand(matrix.shape[0]).to(device)
    lam = 0
    epsilon = 0.001
    N = 1000
    b = 0
    for k in range(N):
        y = x/x.abs().max()
        x = matrix@y
        b = x.max()
        if abs(lam-b)<epsilon:
            return b
        lam = b
    print('err')
    return b
        
    

def calculate_scaled_laplacian(adj, device):
    """
    L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    L' = 2L/lambda - I

    Args:
        adj: adj_matrix

    Returns:
        np.ndarray: L'
    """
    n = adj.shape[0]
    d = torch.sum(adj, axis=1)  # D
    lap = torch.diag(d) - adj     # L=D-A
    # lap1 = lap.clone()
    
    tmp_hm = torch.clamp(d.repeat(n, 1),min=0.0).to(device)
    tmp_mul = tmp_hm * tmp_hm.T
    tmp_sqrt = torch.sqrt(tmp_mul+1e-8)
    tmp_div = torch.clamp(tmp_sqrt,min=1.0).to(device)
    lap /= tmp_div
    
#     for i in range(n):
#         print(i/n)
#         for j in range(n):
#             if d[i] > 0 and d[j] > 0:
#                 lap1[i, j] /= torch.sqrt(d[i] * d[j])
#             if lap[i,j]!=lap1[i,j]:
#                 print(f'error: at {i},{j} {lap[i,j]} != {lap1[i,j]}')
    lap[torch.isinf(lap)] = 0
    lap[torch.isnan(lap)] = 0

    # lap_cpy = lap.detach().cpu().numpy()
    # eigvals = np.linalg.eigvals(lap_cpy).max().real
    # lam = eigvals.max()    
    
    lam = calulate_max_eigval(lap.detach(), device)
    
    return 2 * lap / (lam) - torch.eye(n).to(device) 


def calculate_cheb_poly(lap, ks, device):
    """
    k-order Chebyshev polynomials : T0(L)~Tk(L)
    T0(L)=I/1 T1(L)=L Tk(L)=2LTk-1(L)-Tk-2(L)

    Args:
        lap: scaled laplacian matrix
        ks: k-order

    Returns:
        np.ndarray: T0(L)~Tk(L)
    """
    n = lap.shape[0]
    lap_list = torch.stack([torch.eye(n).to(device), lap[:]]).to(device)
    for i in range(2, ks):
        lap_list = torch.cat([lap_list, torch.stack([torch.matmul(2 * lap, lap_list[-1]).to(device) - lap_list[-2]]).to(device)], dim=0).to(device)
    if ks == 0:
        raise ValueError('Ks must bigger than 0!')
    if ks == 1:
        return lap_list[0:1]  # 1*n*n
    else:
        return lap_list       # Ks*n*n


def calculate_first_approx(weight, device):
    '''
    1st-order approximation function.
    :param W: weighted adjacency matrix of G. Not laplacian matrix.
    :return: np.ndarray
    '''
    # TODO: 如果W对角线本来就是全1？
    n = weight.shape[0]
    adj = weight + torch.diag(torch.ones(n)).to(device)
    # adj = weight + torch.identity(n)
    d = torch.sum(adj, axis=1).to(device)
    # sinvd = np.sqrt(np.mat(np.diag(d)).I)
    # return np.array(sinvd * A * sinvd)
    sinvd = torch.sqrt(torch.linalg.inv(torch.diag(d))).to(device)
    lap = torch.matmul(torch.matmul(sinvd, adj), sinvd).to(device)  # n*n
    lap = torch.unsqueeze(lap, axis=0).to(device)              # 1*n*n
    return lap


class Align(nn.Module):
    def __init__(self, c_in, c_out):
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1)

    def forward(self, x):  # x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  # return: (batch_size, c_out, input_length-1+1, num_nodes-1+1)


class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        """

        :param x: (batch_size, feature_dim(c_in), input_length, num_nodes)
        :return: (batch_size, c_out, input_length-kt+1, num_nodes)
        """
        x_in = self.align(x)[:, :, self.kt - 1:, :]  # (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "GLU":
            # x: (batch_size, c_in, input_length, num_nodes)
            x_conv = self.conv(x)
            # x_conv: (batch_size, c_out * 2, input_length-kt+1, num_nodes)  [P Q]
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
            # return P * sigmoid(Q) shape: (batch_size, c_out, input_length-kt+1, num_nodes)
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)  # residual connection
        return torch.relu(self.conv(x) + x_in)  # residual connection


class SpatioConvLayer(nn.Module):
    def __init__(self, ks, c_in, c_out, device):
        super(SpatioConvLayer, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks).to(device))  # kernel: C_in*C_out*ks
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1).to(device))
        self.align = Align(c_in, c_out)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, lk):
        # Lk: (Ks, num_nodes, num_nodes)
        # x:  (batch_size, c_in, input_length, num_nodes)
        # x_c: (batch_size, c_in, input_length, Ks, num_nodes)
        # theta: (c_in, c_out, Ks)
        # x_gc: (batch_size, c_out, input_length, num_nodes)
        x_c = torch.einsum("knm,bitm->bitkn", lk, x)  # delete num_nodes(n)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta, x_c) + self.b  # delete Ks(k) c_in(i)
        x_in = self.align(x)  # (batch_size, c_out, input_length, num_nodes)
        return torch.relu(x_gc + x_in)  # residual connection


class STConvBlock(nn.Module):
    def __init__(self, ks, kt, n, c, p, device):
        super(STConvBlock, self).__init__()
        self.tconv1 = TemporalConvLayer(kt, c[0], c[1], "GLU")
        self.sconv = SpatioConvLayer(ks, c[1], c[1], device)
        self.tconv2 = TemporalConvLayer(kt, c[1], c[2])
        self.ln = nn.LayerNorm([n, c[2]])
        self.dropout = nn.Dropout(p)

    def forward(self, x, lk):  # x: (batch_size, feature_dim/c[0], input_length, num_nodes)
        x_t1 = self.tconv1(x)    # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_s = self.sconv(x_t1, lk) # (batch_size, c[1], input_length-kt+1, num_nodes)
        x_t2 = self.tconv2(x_s)   # (batch_size, c[2], input_length-kt+1-kt+1, num_nodes)
        x_ln = self.ln(x_t2.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return self.dropout(x_ln)


class FullyConvLayer(nn.Module):
    def __init__(self, c, out_dim):
        super(FullyConvLayer, self).__init__()
        self.conv = nn.Conv2d(c, out_dim, 1)  # c,self.output_dim,1

    def forward(self, x):
        return self.conv(x)


class OutputLayer(nn.Module):
    def __init__(self, c, t, n, out_dim):
        super(OutputLayer, self).__init__()
        self.tconv1 = TemporalConvLayer(t, c, c, "GLU")
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = TemporalConvLayer(1, c, c, "sigmoid")  # kernel=1*1
        self.fc = FullyConvLayer(c, out_dim)

    def forward(self, x):
        # (batch_size, input_dim(c), T, num_nodes)
        x_t1 = self.tconv1(x)
        # (batch_size, input_dim(c), 1, num_nodes)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # (batch_size, input_dim(c), 1, num_nodes)
        x_t2 = self.tconv2(x_ln)
        # (batch_size, input_dim(c), 1, num_nodes)
        return self.fc(x_t2)
        # (batch_size, output_dim, 1, num_nodes)


class mySTGCN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._scaler = self.data_feature.get('scaler')
        self._logger = getLogger()
        
        self.Ks = config.get('Ks', 3)
        self.Kt = config.get('Kt', 3)
        self.blocks = config.get('blocks', [[1, 32, 64], [64, 32, 128]])
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.drop_prob = config.get('dropout', 0)

        self.train_mode = config.get('stgcn_train_mode', 'quick')  # or full
        if self.train_mode.lower() not in ['quick', 'full']:
            raise ValueError('STGCN_train_mode must be `quick` or `full`.')
        self._logger.info('You select {} mode to train STGCN model.'.format(self.train_mode))
        self.blocks[0][0] = self.feature_dim
        if self.input_window - len(self.blocks) * 2 * (self.Kt - 1) <= 0:
            raise ValueError('Input_window must bigger than 4*(Kt-1) for 2 STConvBlock'
                             ' have 4 kt-kernel convolutional layer.')
        self.device = config.get('device', torch.device('cpu'))

        self.graph_conv_type = config.get('graph_conv_type', 'chebconv')

        self.adj_mx = torch.from_numpy(data_feature['adj_mx']).to(self.device)  # torch(ndarray)
        self.lk1 = self.lk2 = None
        self.updateAdj(self.adj_mx, self.adj_mx)

        # 模型结构
        self.st_conv1 = STConvBlock(self.Ks, self.Kt, self.num_nodes,
                                    self.blocks[0], self.drop_prob, self.device)
        self.st_conv2 = STConvBlock(self.Ks, self.Kt, self.num_nodes,
                                    self.blocks[1], self.drop_prob, self.device)
        self.output = OutputLayer(self.blocks[1][2], self.input_window - len(self.blocks) * 2
                                  * (self.Kt - 1), self.num_nodes, self.output_dim)

    def forward(self, batch):
        x = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        x = x.permute(0, 3, 1, 2)  # (batch_size, feature_dim, input_length, num_nodes)
        x_st1 = self.st_conv1(x, self.lk1)   # (batch_size, c[2](64), input_length-kt+1-kt+1, num_nodes)
        x_st2 = self.st_conv2(x_st1, self.lk2)  # (batch_size, c[2](128), input_length-kt+1-kt+1-kt+1-kt+1, num_nodes)
        outputs = self.output(x_st2)  # (batch_size, output_dim(1), output_length(1), num_nodes)
        outputs = outputs.permute(0, 2, 3, 1)  # (batch_size, output_length(1), num_nodes, output_dim)
        return outputs

    def calculate_loss(self, batch):
        if self.train_mode.lower() == 'quick':
            if self.training:  # 训练使用t+1时间步的loss
                y_true = batch['y'][:, 0:1, :, :]  # (batch_size, 1, num_nodes, feature_dim)
                y_predicted = self.forward(batch)  # (batch_size, 1, num_nodes, output_dim)
            else:  # 其他情况使用全部时间步的loss
                y_true = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
                y_predicted = self.predict(batch)  # (batch_size, output_length, num_nodes, output_dim)
        else:   # 'full'
            y_true = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
            y_predicted = self.predict(batch)  # (batch_size, output_length, num_nodes, output_dim)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mse_torch(y_predicted, y_true)

    def predict(self, batch):
        # 多步预测
        x = batch['X']  # (batch_size, input_length, num_nodes, feature_dim)
        y = batch['y']  # (batch_size, output_length, num_nodes, feature_dim)
        y_preds = []
        x_ = x.clone()
        for i in range(self.output_window):
            batch_tmp = {'X': x_}
            y_ = self.forward(batch_tmp)  # (batch_size, 1, num_nodes, output_dim)
            y_preds.append(y_.clone())
            if y_.shape[-1] < x_.shape[-1]:  # output_dim < feature_dim
                y_ = torch.cat([y_, y[:, i:i+1, :, self.output_dim:]], dim=3)
            x_ = torch.cat([x_[:, 1:, :, :], y_], dim=1)
        y_preds = torch.cat(y_preds, dim=1)  # (batch_size, output_length, num_nodes, output_dim)
        return y_preds
    def updateAdj(self, adj_mx1, adj_mx2):
        if self.graph_conv_type.lower() == 'chebconv':
            laplacian_mx = calculate_scaled_laplacian(adj_mx1, self.device)
            self.lk1 = calculate_cheb_poly(laplacian_mx, self.Ks, self.device)
            laplacian_mx2 = calculate_scaled_laplacian(adj_mx2, self.device)
            self.lk2 = calculate_cheb_poly(laplacian_mx2, self.Ks, self.device)
        elif self.graph_conv_type.lower() == 'gcnconv':
            self.lk1 = calculate_first_approx(adj_mx1, self.device)
            self._logger.info('First_approximation_Lk shape: ' + str(lk.shape))
            self.lk2 = calculate_first_approx(adj_mx2, self.device)
            self._logger.info('First_approximation_Lk2 shape: ' + str(lk2.shape))
        else:
            raise ValueError('Error graph_conv_type, must be chebconv or gcnconv.')
