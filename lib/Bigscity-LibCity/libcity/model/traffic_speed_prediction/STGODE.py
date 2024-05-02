import torch
import math
import torch.nn as nn
import numpy as np
from libcity.model import loss
import torch.nn.functional as F
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel

# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

def get_standerd_matrix(M, sigma):
    std = torch.std(M[M != torch.inf])
    mean = torch.mean(M[M != torch.inf])
    M = (M - mean) / std
    return torch.exp(- M**2 / sigma**2)
def get_normalized_adj(A):
    """
    Returns a tensor, the degree normalized adjacency matrix.
    """
    alpha = 0.8
    D = torch.sum(A, axis=1).reshape((-1,))
    D[D <= 10e-5] = 10e-5    # Prevent infs
    diag = torch.reciprocal(torch.sqrt(D))
    A_wave = torch.multiply(torch.multiply(diag.reshape((-1, 1)), A),
                         diag.reshape((1, -1)))
    A_reg = alpha / 2 * (torch.eye(A.shape[0]).to(A_wave.device) + A_wave)
    A_reg[torch.isinf(A_reg)] = 0
    A_reg[torch.isnan(A_reg)] = 0
    return A_reg
# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class ODEFunc(nn.Module):

    def __init__(self, feature_dim, temporal_dim, adj):
        super(ODEFunc, self).__init__()
        self.adj = adj
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)

    def forward(self, t, x):
        torch.enable_grad()
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        xa = torch.einsum('ij, kjlm->kilm', self.adj.float(), x)

        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.einsum('ijkl, lm->ijkm', x, w)

        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        xw2 = torch.einsum('ijkl, km->ijml', x, w2)

        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        return f


class ODEblock(nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1]
        return z


# Define the ODEGCN model.
class ODEG(nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj, time):
        super(ODEG, self).__init__()
        self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim, adj), t=torch.tensor([0, time]))

    def forward(self, x):
        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        return F.relu(z)

class Chomp1d(nn.Module):
    """
    extra dimension will be added by padding, remove it
    """
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()


class TemporalConvNet(nn.Module):
    """
    time dilation convolution
    """
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs : channel's number of input data's feature
            num_channels : numbers of data feature tranform channels, the last is the output channel
            kernel_size : using 1d convolution, so the real kernel is (1, kernel_size) 
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size), padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """ 
        like ResNet
        Args:
            X : input data of shape (B, N, T, F) 
        """
        # permute shape to (B, F, N, T)
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, A_hat):
        """
        Args:
            in_channels: Number of input features at each node in each time step.
            out_channels: a list of feature channels in timeblock, the last is output feature channel
            num_nodes: Number of nodes in the graph
            A_hat: the normalized adjacency matrix
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TemporalConvNet(num_inputs=in_channels,
                                   num_channels=out_channels)
        self.odeg = ODEG(out_channels[-1], 12, A_hat, time=6)
        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1],
                                   num_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X):
        """
        Args:
            X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features)
        Return:
            Output data of shape(batch_size, num_nodes, num_timesteps, out_channels[-1])
        """
        t = self.temporal1(X)
        t = self.odeg(t)
        t = self.temporal2(F.relu(t))
        return self.batch_norm(t)


class STGODE(AbstractTrafficStateModel):
    """ the overall network framework """
    def __init__(self, config, data_feature):
        """ 
        Args:
            num_nodes : number of nodes in the graph
            num_features : number of features at each node in each time step
            input_window : number of past time steps fed into the network
            output_window : desired number of future time steps output by the network
            A_sp_hat : nomarlized adjacency spatial matrix
            A_se_hat : nomarlized adjacency semantic matrix
        """
        super().__init__(config, data_feature)
        # section 1: data_feature
        self._scaler = self.data_feature.get('scaler')
        self.num_nodes = self.data_feature.get('num_nodes', 1)
        self.feature_dim = self.data_feature.get('feature_dim', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self._logger = getLogger()
        # section 2: model config
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.device = config.get('device', torch.device('cpu'))
        # self.hidden_size = config.get('hidden_size', 64)
        # self.num_layers = config.get('num_layers', 1)
        # self.dropout = config.get('dropout', 0)
        """
        Args:
        sigma1: float, default=0.1, sigma for the semantic matrix
        sigma2: float, default=10, sigma for the spatial matrix
        thres1: float, default=0.6, the threshold for the semantic matrix
        thres2: float, default=0.5, the threshold for the spatial matrix
        """
        self.sigma1 = config.get('sigma1', 0.1)
        self.sigma2 = config.get('sigma2', 10)
        self.thres1 = config.get('thres1', 0.6)
        self.thres2 = config.get('thres2', 0.5)
 
        self.adj_mx = torch.tensor(self.data_feature['adj_mx']).to(self.device)
        self.A_sp_hat = get_normalized_adj(self.adj_mx)

        self.dist_mx = get_standerd_matrix(torch.tensor(self.data_feature['dist_mx']).to(self.device), self.sigma1)
        self.dtw_mx = torch.zeros_like(self.dist_mx)
        self.dtw_mx[self.dist_mx > self.thres1] = 1
        self.A_se_hat = get_normalized_adj(self.dtw_mx)
        # section 3: model structure
        # spatial graph
        self.sp_blocks = nn.ModuleList(
            [nn.Sequential(
                STGCNBlock(in_channels=self.feature_dim, out_channels=[64, 32, 64],
                num_nodes=self.num_nodes, A_hat=self.A_sp_hat),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                num_nodes=self.num_nodes, A_hat=self.A_sp_hat)) for _ in range(3)
            ])
        # semantic graph
        self.se_blocks = nn.ModuleList([nn.Sequential(
                STGCNBlock(in_channels=self.feature_dim, out_channels=[64, 32, 64],
                num_nodes=self.num_nodes, A_hat=self.A_se_hat),
                STGCNBlock(in_channels=64, out_channels=[64, 32, 64],
                num_nodes=self.num_nodes, A_hat=self.A_se_hat)) for _ in range(3)
            ]) 

        self.pred = nn.Sequential(
            nn.Linear(self.input_window * 64, self.output_window * 32), 
            nn.ReLU(),
            nn.Linear(self.output_window * 32, self.output_window * self.output_dim)
        )

    def forward(self, batch):
        x = batch['X'] # (batch_size, input_length, num_nodes, feature_dim)
        x = x.permute(0, 2, 1, 3)  # (batch_size, num_nodes, input_length, feature_dim)
        """
        Args:
            x : input data of shape (batch_size, num_nodes, num_timesteps, num_features) == (B, N, T, F)
        Returns:
            prediction for future of shape (batch_size, num_nodes, output_window, output_dim)
        """
        outs = []
        # spatial graph
        for blk in self.sp_blocks:
            outs.append(blk(x))
        # semantic graph
        for blk in self.se_blocks:
            outs.append(blk(x))
        outs = torch.stack(outs)
        x = torch.max(outs, dim=0)[0]
        x = x.reshape((x.shape[0], x.shape[1], -1))
        return self.pred(x).reshape(x.shape[0],x.shape[1],self.output_window,-1).permute(0, 2, 1, 3)

    def predict(self, batch):
        return self.forward(batch)

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.smooth_l1_loss(y_predicted, y_true)

    def updateAdj(self, adj_mx1, adj_mx2):
        A_sp_hat1 = get_normalized_adj(adj_mx1)
        A_sp_hat2 = get_normalized_adj(adj_mx2)

        for seq in self.sp_blocks:
            seq[0].odeg.odeblock.odefunc.adj=A_sp_hat1
            seq[1].odeg.odeblock.odefunc.adj=A_sp_hat2