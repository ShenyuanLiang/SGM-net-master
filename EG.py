# %%
import code
import os
from glob import glob

from numpy import float32
import sys

import numpy as np
import torch

from models.BBGM.affinity_layer import InnerProductWithWeightsAffinity
from models.BBGM.sconv_archs import (SiameseNodeFeaturesToEdgeFeatures,
                                     SiameseSConvOnNodes)
from src1.utils.config import cfg, cfg_from_file

from src1.lap_solvers.sinkhorn import GumbelSinkhorn, Sinkhorn

cfg_from_file('experiments/ngm_qaplib.yaml')
import sys
from tqdm import tqdm
from src1.backbone import *
from src1.evaluation_metric import objective_score, objective_score_node, objective_score_edge, Node_distance, \
    Edge_distance

from src1.lap_solvers.hungarian import hungarian
import scipy.sparse as ssp
from torch.utils.tensorboard import SummaryWriter
sys.path.append('new_code/src')
from torch.utils.data import DataLoader, Dataset
from new_code.src.model import (get_criterion, get_optimizer, get_scheduler,
                                matching_accuracy)
# %%
from new_code.src.dataset import GMDataset
from Shape.Statistics import *
from torch_sparse import SparseTensor
import torch.nn.functional as F
import argparse
from sklearn.metrics.pairwise import cosine_similarity, rbf_kernel

###########################################################

parser = argparse.ArgumentParser(description="SGM-v2 for shape graph matching")


# ================================================================================================================ #
parser.add_argument('--exp-model', type=str, \
                    required=False, help="set experiment's model dir (created by `train.py`)")

parser.add_argument('--train-device', type=str, default="cuda", help='training device, cuda or cpu')
parser.add_argument('--test-device', type=str, default="cpu", help='Inference device, cuda or cpu')
parser.add_argument('--exp-traind', type=str, \
                    required=False, help="set experiment's train dataset dir (created by `train.py`)")

parser.add_argument('--exp-evald', type=str, \
                    required=False, help="set experiment's test dataset dir (created by `train.py`)")

parser.add_argument('--exp-savep', type=str, \
                    required=False, help="set experiment's permutation save dir (created by `train.py`)")
parser.add_argument('--file-name', type=str, \
                    required=False, help="set experiment's permutation file name (created by `train.py`)")

parser.add_argument('--checkpoint-mod', type=str, \
                    required=False, help="check point path for model")

parser.add_argument('--checkpoint-opt', type=str, \
                    required=False, help="check point path for optimizer")

parser.add_argument('--tensorboard-savep', type=str, \
                    required=False, help="tensorboard log file save dir (created by `train.py`)")
parser.add_argument('--lr', type=float, default=1.e-3, help="set learning rate")
parser.add_argument('--Inference', action='store_true')
parser.add_argument('--Resume-training', action='store_true', help='resume training from a check point')
parser.add_argument('--Train', action='store_true')
parser.add_argument('--Mean', action='store_true')
parser.add_argument('--ground-truth', action='store_true', help='whether has ground truth information')
parser.add_argument('--weights-on-edges', type=float, default=1, help="set weight parameter on edge affinity matrix")
# args = parser.parse_args(args=[])
args, _ = parser.parse_known_args()
# args = parser.parse_args()

if args.tensorboard_savep is not None:
    os.makedirs(args.tensorboard_savep, exist_ok=True)
    writer = SummaryWriter(args.tensorborad)


class SHAPEGRAPH(Dataset):
    def __init__(self, path: str, inference=False):
        self.inference = inference

        data_path = os.path.join(path, '*t')
        files = glob(data_path)
        # EG1_all = []
        # EG2_all = []
        perm_mat = []
        file_name = []
        Ke = []
        Kp = []
        node_attr1, node_attr2, edge_attr1, edge_attr2 = [], [], [], []
        H1 = []
        H2 = []
        G1 = []
        G2 = []
        A1, A2 = [], []

        for f1 in tqdm(files):
            data = loadmat(f1)
            pair = data['pair']
            EG = [pair['G1'], pair['G2']]
            for i in range(len(EG)):
                EG[i] = {key: np.asarray(value, dtype=np.float32) for key, value in EG[i].items()}
            # EG[0],_ = addOneWayNullNodes(EG[0], EG[1])
            # EG1_all.append(EG[0])
            # EG2_all.append(EG[1])
            n1 = np.asarray(EG[0]['A']).shape[0]
            n2 = np.asarray(EG[1]['A']).shape[0]
            if n1 > n2:
                raise Exception('EG1 should be smaller than EG2')
            permutation_vector = np.asarray(pair['Perm_v']) - 1
            permutation_matrix = np.zeros((n1, n2), dtype=np.int32)
            for i, j in enumerate(permutation_vector):
                permutation_matrix[i, j] = 1
            perm_mat.append(permutation_matrix)
            file_name.append(f1)
            node_attr1.append(np.asarray(EG[0]['nodeXY'], dtype=np.float32))
            node_attr2.append(np.asarray(EG[1]['nodeXY'], dtype=np.float32))
            edge_attr1.append(np.asarray(EG[0]['Abetaq'], dtype=np.float32))
            edge_attr2.append(np.asarray(EG[1]['Abetaq'], dtype=np.float32))
            if self.inference:
                Ke.append(np.asarray(pair['Kq'], dtype=np.float32))
                Kp.append(np.asarray(pair['Kp'], dtype=np.float32))
            H1.append(np.asarray(EG[0]['H'], dtype=np.float32))
            H2.append(np.asarray(EG[1]['H'], dtype=np.float32))
            G1.append(np.asarray(EG[0]['G'], dtype=np.float32))
            G2.append(np.asarray(EG[1]['G'], dtype=np.float32))
            A1.append(np.asarray(EG[0]['A'], dtype=np.float32))
            A2.append(np.asarray(EG[1]['A'], dtype=np.float32))
            # self.P, self.file_name, \
            #      self.Kp, self.Ke, self.H1, \
            #          self.H2, self.G1, \
            #              self.G2 = perm_mat, file_name, Kp, Ke, H1, H2, G1,G2
            if not self.inference:
                self.P, self.file_name, \
                self.node_attr1, self.node_attr2, self.edge_attr1, self.edge_attr2, self.H1, \
                self.H2, self.G1, \
                self.G2, self.A1, self.A2 = perm_mat, file_name, node_attr1, node_attr2, \
                                            edge_attr1, edge_attr2, H1, H2, G1, G2, A1, A2

            else:
                self.P, self.file_name, \
                self.node_attr1, self.node_attr2, self.edge_attr1, self.edge_attr2, self.H1, \
                self.H2, self.G1, \
                self.G2, self.Ke, self.Kp, self.A1, self.A2 = perm_mat, file_name, node_attr1, node_attr2, \
                                                              edge_attr1, edge_attr2, H1, H2, G1, G2, Ke, Kp, A1, A2

        self.n_samples = len(perm_mat)
        # here the first column is the class label, the rest are the features

    def __getitem__(self, idx):
        # EG1 = self.EG1[idx]
        # EG2 = self.EG2[idx]
        # Ke, Kp, G1, G2, H1, H2 = self.Ke[idx], self.Kp[idx], self.G1[idx], \
        #     self.G2[idx], self.H1[idx], self.H2[idx]
        if not self.inference:
            node_attr1, node_attr2, edge_attr1, edge_attr2, G1, G2, H1, H2, A1, A2, Kp, Ke = self.node_attr1[idx], \
                                                                                             self.node_attr2[idx], \
                                                                                             self.edge_attr1[idx], \
                                                                                             self.edge_attr2[idx], \
                                                                                             self.G1[idx], \
                                                                                             self.G2[idx], self.H1[idx], \
                                                                                             self.H2[idx], \
                                                                                             self.A1[idx], self.A2[idx], \
                                                                                             self.Kp[idx], self.Ke[idx]

            data_dict = {
                'file_name': self.file_name[idx],
                'Kp': torch.from_numpy(Kp),
                'Ke': torch.from_numpy(Ke),
                'node_attr1': torch.from_numpy(node_attr1),
                'node_attr2': torch.from_numpy(node_attr2),
                'edge_attr1': torch.from_numpy(edge_attr1),
                'edge_attr2': torch.from_numpy(edge_attr2),
                'H': [np.asarray(x) for x in [H1, H2]],
                'G': [np.asarray(x) for x in [G1, G2]],
                'A1': torch.from_numpy(A1),
                'A2': torch.from_numpy(A2),
                'permutation_matrix': torch.from_numpy(self.P[idx])
            }
        else:
            node_attr1, node_attr2, edge_attr1, edge_attr2, G1, G2, H1, H2, A1, A2, Kp, Ke = self.node_attr1[idx], \
                                                                                             self.node_attr2[idx], \
                                                                                             self.edge_attr1[idx], \
                                                                                             self.edge_attr2[idx], \
                                                                                             self.G1[idx], \
                                                                                             self.G2[idx], self.H1[idx], \
                                                                                             self.H2[idx], \
                                                                                             self.A1[idx], self.A2[idx], \
                                                                                             self.Kp[idx], self.Ke[idx]
            data_dict = {
                'file_name': self.file_name[idx],
                'Kp': torch.from_numpy(Kp),
                'Ke': torch.from_numpy(Ke),
                'node_attr1': torch.from_numpy(node_attr1),
                'node_attr2': torch.from_numpy(node_attr2),
                'edge_attr1': torch.from_numpy(edge_attr1),
                'edge_attr2': torch.from_numpy(edge_attr2),
                'H': [np.asarray(x) for x in [H1, H2]],
                'G': [np.asarray(x) for x in [G1, G2]],
                'A1': torch.from_numpy(A1),
                'A2': torch.from_numpy(A2),
                'permutation_matrix': torch.from_numpy(self.P[idx])
            }
        return data_dict

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


def kronecker_sparse(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    r"""
    Compute the kronecker product of :math:`\mathbf{T}_1` and :math:`\mathbf{T}_2`.
    This function is implemented in scipy.sparse API and runs on cpu.

    :param arr1: input array 1
    :param arr2: input array 2
    :return: kronecker product of :math:`\mathbf{T}_1` and :math:`\mathbf{T}_2`
    """
    s1 = ssp.coo_matrix(arr1)
    s2 = ssp.coo_matrix(arr2)
    ss = ssp.kron(s1, s2)
    return ss


class GNNLayer(nn.Module):
    def __init__(self, in_node_features, in_edge_features, out_node_features, out_edge_features,
                 sk_channel=0, sk_iter=20, sk_tau=0.05, edge_emb=False):
        super(GNNLayer, self).__init__()
        self.in_nfeat = in_node_features
        self.in_efeat = in_edge_features
        self.out_efeat = out_edge_features
        self.sk_channel = sk_channel
        assert out_node_features == out_edge_features + self.sk_channel
        if self.sk_channel > 0:
            self.out_nfeat = out_node_features - self.sk_channel
            self.sk = Sinkhorn(sk_iter, sk_tau)
            self.classifier = nn.Linear(self.out_nfeat, self.sk_channel)
        else:
            self.out_nfeat = out_node_features
            self.sk = self.classifier = None

        if edge_emb:
            self.e_func = nn.Sequential(
                nn.Linear(self.in_efeat + self.in_nfeat, self.out_efeat),
                nn.ReLU(),
                nn.Linear(self.out_efeat, self.out_efeat),
                nn.ReLU()
            )
        else:
            self.e_func = None

        if edge_emb:
            self.e_func2 = nn.Sequential(
                nn.Linear(self.in_efeat + self.in_nfeat, self.out_efeat),
                nn.ReLU(),
                nn.Linear(self.out_efeat, self.out_efeat),
                nn.ReLU(),
            )
        else:
            self.e_func2 = None

        self.n_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            # nn.Linear(self.in_nfeat, self.out_nfeat // self.out_efeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            # nn.Linear(self.out_nfeat // self.out_efeat, self.out_nfeat // self.out_efeat),
            nn.ReLU(),
        )

        self.n_self_func = nn.Sequential(
            nn.Linear(self.in_nfeat, self.out_nfeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat, self.out_nfeat),
            nn.ReLU()
        )
        self.combine_func = nn.Sequential(
            nn.Linear(self.out_nfeat + self.out_efeat, self.out_nfeat + self.out_efeat),
            nn.ReLU(),
            nn.Linear(self.out_nfeat + self.out_efeat, self.out_nfeat),
            nn.ReLU()
        )


    def forward(self, A, W, x, n1=None, n2=None, norm=False):
        """
        :param A: adjacent matrix in 0/1 (b x n x n)
        :param W: edge feature tensor (b x n x n x feat_dim)
        :param x: node feature tensor (b x n x feat_dim)
        """
        if self.e_func is not None:
            W1 = A.matmul(x)
            # W1 = x
            # W2 = torch.cat((A.matmul(W), W1), dim=-1)
            W2 = torch.cat((W, W1), dim=-1)
            # W2 = W
            W_new1 = self.e_func(W2)
        else:
            W_new1 = W

        if norm is True:
            A = F.normalize(A.to_dense(), p=1, dim=1)


        x2 = torch.cat((W_new1, self.n_self_func(x)), dim=-1)
        x2 = self.combine_func(x2)


        ############################################################### update
        # W_new = self.combine_func3_(torch.cat((W_new1, x4), dim=1))
        W_new = W_new1
        # x5 = self.combine_func3(torch.cat((x2, W_new2), dim=1))
        x5 = x2

        if self.classifier is not None:
            assert n1.max() * n2.max() == x.shape[1]
            x3 = self.classifier(x2)
            n1_rep = torch.repeat_interleave(n1, self.sk_channel, dim=0)
            n2_rep = torch.repeat_interleave(n2, self.sk_channel, dim=0)
            x4 = x3.permute(0, 2, 1).reshape(x.shape[0] * self.sk_channel, n2.max(), n1.max()).transpose(1, 2)
            x5 = self.sk(x4, n1_rep, n2_rep, dummy_row=True).transpose(2, 1).contiguous()

            x6 = x5.reshape(x.shape[0], self.sk_channel, n1.max() * n2.max()).permute(0, 2, 1)
            x_new = torch.cat((x2, x6), dim=-1)
        else:
            x_new = x5

        return W_new, x_new


import math


def calculate_output_length(length_in, kernel_size, stride=1, padding=0, dilation=1):
    return (length_in + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


def adj2GH(A):
    n = int(A.shape[0])
    m = 0
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0.:
                m += 1
    m = int(m)
    # m = int(np.sum(A, axis=(0, 1)))
    G = np.zeros((n, m), dtype=np.float32)
    H = np.zeros_like(G)
    edge_idx = 0
    for i in range(n):
        range_j = range(n)
        for j in range_j:
            if A[i, j] > 0.:
                G[i, edge_idx] = 1
                # G[j, edge_idx+int(m/2)] = 1
                H[j, edge_idx] = 1
                # H[i, edge_idx+int(m/2)] = 1
                edge_idx += 1
    return (torch.from_numpy(G).unsqueeze(0), torch.from_numpy(H).unsqueeze(0))


class Affinity(nn.Module):
    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        self.A = nn.parameter.Parameter(torch.Tensor(self.d, self.d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d)

    def forward(self, X, Y):
        assert X.shape[2] == Y.shape[2] == self.d
        M = torch.matmul(X, (self.A + self.A.transpose(0, 1)) / 2)
        M = torch.matmul(M, Y.transpose(1, 2))
        return M


def pairwise_euclidean_similarity(x, y):
    """Compute the pairwise Euclidean similarity between x and y.
    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = -|x_i - y_j|^2.
    Args:
      x: NxD float tensor.
      y: MxD float tensor.
    Returns:
      s: NxM float tensor, the pairwise euclidean similarity.
    """
    distM = torch.cdist(x, y, p=2)
    sim = 1 - distM / (distM.max())
    return sim




def pairwise_cosine_similarity(x, y):
    """Compute the cosine similarity between x and y.
    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = x_i^T y_j / (|x_i||y_j|).
    Args:
      x: NxD float tensor.
      y: MxD float tensor.
    Returns:
      s: NxM float tensor, the pairwise cosine similarity.
    """
    a_norm = x / x.norm(dim=2)[:, :, None]
    b_norm = y / y.norm(dim=2)[:, :, None]
    res = torch.matmul(a_norm, b_norm.transpose(1, 2))

    return res


def pairwise_euclidean_similarity1(x, y):
    """Compute the pairwise Euclidean similarity between x and y.
    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = -|x_i - y_j|^2.
    Args:
      x: NxD float tensor.
      y: MxD float tensor.
    Returns:
      s: NxM float tensor, the pairwise euclidean similarity.
    """
    s = 2 * torch.mm(x, torch.transpose(y, 1, 0))
    diag_x = torch.sum(x * x, dim=-1)
    diag_x = torch.unsqueeze(diag_x, 0)
    diag_y = torch.reshape(torch.sum(y * y, dim=-1), (1, -1))

    return s - diag_x - diag_y


class Edge_compose(nn.Module):
    def __init__(self, in_edge_features, hidden_features, out_edge_features):
        super(Edge_compose, self).__init__()
        self.in_efeat = in_edge_features
        self.hidden_features = hidden_features
        self.out_efeat = out_edge_features

        self.conv1 = nn.Conv1d(self.in_efeat, self.hidden_features, 15)
        self.conv2 = nn.Conv1d(self.hidden_features, 2 * self.hidden_features, 16)

        self.fc1 = nn.Linear(2 * self.hidden_features, self.hidden_features)
        self.fc2 = nn.Linear(self.hidden_features, self.out_efeat)
        # self.fc2 = nn.Linear(2 * self.hidden_features, out_edge_features)

    def forward(self, edge_attr):
        # x = self.conv1(edge_attr)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = x.view(x.shape[0], -1)
        x = self.fc1(edge_attr)
        x = F.relu(x)
        x = self.fc2(x)
        return F.sigmoid(x)


class edge_passing(nn.Module):
    def __init__(self, in_edge_features, hidden_features, out_edge_features):
        super(edge_passing, self).__init__()
        self.in_efeat = in_edge_features
        self.hidden_features = hidden_features
        self.out_efeat = out_edge_features

        self.conv1 = nn.Conv1d(self.in_efeat, self.hidden_features, 15)
        self.conv2 = nn.Conv1d(self.hidden_features, 2 * self.hidden_features, 16)

        self.fc1 = nn.Linear(2 * self.hidden_features, self.hidden_features)
        self.fc2 = nn.Linear(self.hidden_features, self.out_efeat)
        # self.fc2 = nn.Linear(2 * self.hidden_features, out_edge_features)

    def forward(self, edge_attr):
        x = self.conv1(edge_attr)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.sigmoid(x)


class node_passing(nn.Module):
    def __init__(self, in_node_features, hidden_features, out_node_features):
        super(node_passing, self).__init__()
        self.in_nfeat = in_node_features
        self.hidden_features = hidden_features
        self.out_nfeat = out_node_features

        self.fc1 = nn.Linear(self.in_nfeat, self.hidden_features)
        self.fc2 = nn.Linear(self.hidden_features, self.out_nfeat)

    def forward(self, node_attr):
        x = self.fc1(node_attr)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.normalize(self.fc2(x), dim=-1, p=2)
        return x




from torch_geometric.nn import GATv2Conv, SAGEConv


class get_graph_embedding(torch.nn.Module):
    def __init__(self):
        super(get_graph_embedding, self).__init__()
        self.edge_fun = edge_passing(2, 30, 60)
        # self.conv = SplineConv(2, 2, dim=5, kernel_size=5, aggr="max")
        self.conv2 = GATv2Conv(60, 60, edge_dim=60)
        # self.node_passing = node_passing(2, 60, 60)
        self.node_passing = SAGEConv(2, 60, project=True, normalize=True)
        self.edge_passing = nn.Sequential(
            nn.Linear(60, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512)
        )
        # self.node_passing = ClusterGCNConv(2, 128)
        # self.conv3 = NNConv(5, 30, nn=nn.Sequential(
        #     nn.Linear(60, int(5 * 30 / 2)),
        #     nn.ReLU(),
        #     nn.Linear(int(5 * 30 / 2), 5 * 30),
        #     nn.ReLU()
        # ))
        # self.conv3 = HEATConv(128, 256, num_node_types=1, num_edge_types=10,
        #                       edge_type_emb_dim=20, edge_dim=60,
        #                       edge_attr_emb_dim=256)

    def forward(self, node_attr1, n_points, G1, H1):
        device = node_attr1.device
        A1 = (G1 @ H1.permute(0, -1, -2)).to(device)
        index_edge = torch.argwhere(A1[0] != 0).T

        # edge_attr1 = edge_attr1.permute(0, -1, -2, -3, -4)
        # edge_attr1 = edge_attr1[:, index_edge[0, :], index_edge[1, :], :, :]
        # edge_output = edge_fun(edge_attr1.transpose(2, 3).squeeze(0))

        # edge_emb_fun = edge_passing(2, 30, 5).to(device)
        # edge_emb = self.edge_fun(edge_attr1.transpose(2, 3).squeeze(0))
        # edge_emb = torch.flatten(edge_attr1, start_dim=2, end_dim=3)
        # edge_output = F.sigmoid(edge_attr1)
        # edge_output = self.edge_passing(edge_emb)
        # node_layer = node_passing(2, 10, 30).to(device)
        # node_emb = node_layer(node_attr1.permute(0, 2, 1))
        node_embedding = self.node_passing(node_attr1, index_edge)
        #
        # out = self.conv2(node_embedding[0], index_edge, edge_emb[0])
        # node_emb = node_embedding + .1 * out
        return node_embedding, device


CNN = eval(cfg.BACKBONE)


class Net(CNN):
    def __init__(self):
        super(Net, self).__init__()
        self.message_pass_node_features = SiameseSConvOnNodes(input_node_dim=cfg.NGM.FEATURE_CHANNEL * 2)
        self.build_edge_features_from_node_features = SiameseNodeFeaturesToEdgeFeatures(
            total_num_nodes=self.message_pass_node_features.num_node_features
        )
        self.global_state_dim = cfg.NGM.FEATURE_CHANNEL * 2

        self.GATconv = GATv2Conv(1, 30, edge_dim=1)
        self.GATconv2 = GATv2Conv(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB,
                                  cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, edge_dim=1)



        self.rescale = cfg.PROBLEM.RESCALE
        self.tau = cfg.NGM.SK_TAU
        self.mgm_tau = cfg.NGM.MGM_SK_TAU
        self.univ_size = cfg.NGM.UNIV_SIZE

        self.sinkhorn = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau, epsilon=cfg.NGM.SK_EPSILON)
        self.sinkhorn_mgm = Sinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, epsilon=cfg.NGM.SK_EPSILON, tau=self.mgm_tau)
        self.gumbel_sinkhorn = GumbelSinkhorn(max_iter=cfg.NGM.SK_ITER_NUM, tau=self.tau * 10,
                                              epsilon=cfg.NGM.SK_EPSILON, batched_operation=True)
        self.l2norm = nn.LocalResponseNorm(cfg.NGM.FEATURE_CHANNEL * 2, alpha=cfg.NGM.FEATURE_CHANNEL * 2, beta=0.5,
                                           k=0)
        self.gnn_layer = cfg.NGM.GNN_LAYER


        for i in range(self.gnn_layer):
            tau = cfg.NGM.SK_TAU
            if i == 0:
                gnn_layer = GNNLayer(30, 1,
                                     cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=True)  # cfg.NGM.EDGE_EMB
                GATconv_layer = GATv2Conv(30, 30, edge_dim=1)
            else:
                gnn_layer = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                                     cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                     sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=True)  # cfg.NGM.EDGE_EMB

                GATconv_layer = GATv2Conv(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB,
                                          cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, edge_dim=1)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('gat_layer_{}'.format(i), GATconv_layer)

        self.gnn_layer_edge = cfg.NGM.GNN_LAYER
        for i in range(self.gnn_layer_edge):
            tau = cfg.NGM.SK_TAU
            if i == 0:
                gnn_layer_edge = GNNLayer(1, 1,
                                          cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                          sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=True)  # cfg.NGM.EDGE_EMB
            else:
                gnn_layer_edge = GNNLayer(cfg.NGM.GNN_FEAT[i - 1] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i - 1],
                                          cfg.NGM.GNN_FEAT[i] + cfg.NGM.SK_EMB, cfg.NGM.GNN_FEAT[i],
                                          sk_channel=cfg.NGM.SK_EMB, sk_tau=tau, edge_emb=True)  # cfg.NGM.EDGE_EMB
            self.add_module('gnn_layer_edge_{}'.format(i), gnn_layer_edge)

        # self.classifier = nn.Linear(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, 1)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, (cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB) // 2),
            # nn.ReLU(),
            nn.Linear((cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB) // 2, (cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB) // 4),
            # nn.ReLU(),
            nn.Linear((cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB) // 4, 1)
        )
        self.classifier_edge = nn.Sequential(
            nn.Linear(cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB, (cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB) // 2),
            # nn.ReLU(),
            nn.Linear((cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB) // 2, (cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB) // 4),
            # nn.ReLU(),
            nn.Linear((cfg.NGM.GNN_FEAT[-1] + cfg.NGM.SK_EMB) // 4, 1)
        )


    def forward(self, Kp, Ke, G1, G2, H1, H2, \
                n_points, idx1=0, idx2=1):

        # Kp = Kp * 400
        Ke = Ke / args.weights_on_edges
        device = Ke.device


        coo = kronecker_sparse(G1[0].numpy() @ H1[0].numpy().T, G2[0].numpy() @ H2[0].numpy().T)
        A = SparseTensor.from_scipy(coo).to(device)
        # emb_node = torch.log(Kp.view(-1).unsqueeze(-1) + 0.01)
        # emb_node += torch.abs(emb_node.min())
        emb_node = Kp.view(-1).unsqueeze(-1) * 2
        #
        emb_edge = torch.matmul(torch.matmul(G1[0].to(device), Ke[0]), G2[0].to(device).T) * \
                   torch.matmul(torch.matmul(H1[0].to(device), Ke[0]), H2[0].to(device).T)

        #############################################################################
        emb_edge = torch.log(emb_edge.view(-1).unsqueeze(-1) + 0.01)
        emb_edge += torch.abs(emb_edge.min())
        emb_edge = (emb_edge / emb_edge.max())

        #############################################################################
        ####### Gatconv
        row, col, _ = A.t().coo()
        edge_index = torch.stack([row, col], dim=0)
        edge_emb_ = Ke.view(-1).unsqueeze(-1)
        # emb_node = self.Graphsage(emb_node, edge_index)
        emb_node = self.GATconv(emb_node, edge_index, edge_emb_)

        #####################################################################
        # emb_edge2 = edge_emb_.clone()
        # coo = kronecker_sparse(G1[0].numpy().T @ H1[0].numpy(), G2[0].numpy().T @ H2[0].numpy())
        # A2 = SparseTensor.from_scipy(coo).to(device)
        # emb_node2 = torch.matmul(torch.matmul(G1[0].to(device).T, Kp[0]), G2[0].to(device)) * \
        #             torch.matmul(torch.matmul(H1[0].to(device).T, Kp[0]), H2[0].to(device))
        # emb_node2 = emb_node2.view(-1).unsqueeze(-1)
        #####################################################################

        # NGM qap solver
        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            # gat_layer = getattr(self, 'gat_layer_{}'.format(i))
            # emb_node = gat_layer(emb_node, edge_index, edge_emb_)
            emb_edge, emb_node = gnn_layer(A, emb_edge, emb_node, n_points[idx1], n_points[idx2])



        v = self.classifier(emb_node)
        v_edge = self.classifier_edge(emb_edge)
        s = v.reshape(n_points[idx1], n_points[idx2]).unsqueeze(0)
        s_edge = v_edge.reshape(n_points[idx1], n_points[idx2]).unsqueeze(0)
        s = (s + s_edge) / 2

        if self.training or cfg.NGM.GUMBEL_SK <= 0:
            # if cfg.NGM.GUMBEL_SK <= 0:
            ss = self.sinkhorn(s, n_points[idx1], n_points[idx2], dummy_row=True)
        else:
            gumbel_sample_num = cfg.NGM.GUMBEL_SK
            if self.training:
                gumbel_sample_num //= 10
            ss_gumbel = self.gumbel_sinkhorn(s, n_points[idx1], n_points[idx2], sample_num=gumbel_sample_num,
                                             dummy_row=True)

            repeat = lambda x, rep_num=gumbel_sample_num: torch.repeat_interleave(x, rep_num, dim=0)
            if not self.training:
                ss_gumbel = hungarian(ss_gumbel, repeat(n_points[idx1]), repeat(n_points[idx2]))
            ss_gumbel = ss_gumbel.reshape(1, gumbel_sample_num, ss_gumbel.shape[-2], ss_gumbel.shape[-1])


            max_repeats = gumbel_sample_num
            obj_score = []
            for idx in range(0, gumbel_sample_num, max_repeats):
                if idx + max_repeats > gumbel_sample_num:
                    rep_num = gumbel_sample_num - idx
                else:
                    rep_num = max_repeats

                obj_score.append(
                    (objective_score_edge(
                        ss_gumbel[:, idx:(idx + rep_num), :, :].reshape(-1, ss_gumbel.shape[-2], ss_gumbel.shape[-1]),
                        repeat(Kp, rep_num), repeat(Ke, rep_num), G1.to(device),
                        G2.to(device), H1.to(device), H2.to(device)

                    ) + objective_score_node(
                        ss_gumbel[:, idx:(idx + rep_num), :, :].reshape(-1, ss_gumbel.shape[-2], ss_gumbel.shape[-1]),
                        repeat(Kp, rep_num), repeat(Ke, rep_num), G1.to(device),
                        G2.to(device), H1.to(device), H2.to(device)

                    )).reshape(1, -1)
                )
            obj_score = torch.cat(obj_score, dim=1)
            min_obj_score = obj_score.min(dim=1)
            ss = ss_gumbel[torch.arange(1), min_obj_score.indices.to(device), :, :]

        return ss, Kp, Ke




from pathlib import Path
import time
from new_code.src.config import cfg as CFG


def eval_model(model, dataloader, visualize=False, dst=''):
    model.eval()
    device = torch.device(args.test_device)
    print(f'inference is done in {device}')
    since = time.time()
    acc_match_num = torch.zeros(1).to(device)
    acc_total_num = torch.zeros(1).to(device)
    obj = torch.zeros(1).to(device)
    obj_mat = torch.zeros_like(obj).to(device)
    obj_org = torch.zeros(1).to(device)
    obj_mat_node = torch.zeros_like(obj).to(device)
    for inputs in dataloader:
        perm_mat = inputs['permutation_matrix'].float().to(device)
        n1_gt, n2_gt = [_.to(device) for _ in
                        [torch.Tensor([perm_mat.shape[1]]).int(), torch.Tensor([perm_mat.shape[2]]).int()]]
        n_points = [n1_gt, n2_gt]
        Kp = inputs['Kp'].float().to(device)
        Ke = inputs['Ke'].float().to(device) * args.weights_on_edges

        G1, G2 = [_ for _ in inputs['G']]
        H1, H2 = [_ for _ in inputs['H']]


        with torch.set_grad_enabled(False):
            s_pred, _, _ = model(Kp, Ke,
                                 G1, G2, H1, H2, n_points)

        s_pred_perm = hungarian(s_pred, n1_gt, n2_gt)
        _obj = -1 * objective_score_edge(s_pred_perm, Kp, Ke, G1.to(device), G2.to(device), H1.to(device),
                                         H2.to(device))
        obj += _obj
        _obj_mat = -1 * objective_score_edge(perm_mat, Kp, Ke, G1.to(device), G2.to(device), H1.to(device),
                                             H2.to(device))
        obj_mat += _obj_mat
        ################################################################
        obj_org_ = -1 * objective_score_node(s_pred_perm, Kp, Ke, G1.to(device), G2.to(device), H1.to(device),
                                             H2.to(device))
        obj_org += obj_org_
        _obj_mat_node = -1 * objective_score_node(perm_mat, Kp, Ke, G1.to(device), G2.to(device), H1.to(device),
                                                  H2.to(device))
        obj_mat_node += _obj_mat_node

        _, _acc_match_num, _acc_total_num = matching_accuracy(s_pred_perm, perm_mat, n1_gt)
        acc_match_num += _acc_match_num
        acc_total_num += _acc_total_num
    print(f'edge objective ratio [network / matlab] is: {(obj / obj_mat).item()}')
    print(f'node objective ratio [network / matlab] is: {(obj_org / obj_mat_node).item()}')
    print(f'objective score is: {(obj / len(dataloader.dataset)).item()}')

    print('ACC = {:.4f} ({}/{}) in {:.2f} seconds'.format((acc_match_num / acc_total_num).item(),
                                                          int(acc_match_num.item()), int(acc_total_num.item()),
                                                          time.time() - since))

    return obj / obj_mat


def train_eval_model(model, dataloader_train, dataloader_test):
    device = torch.device(args.train_device)
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    if not args.ground_truth:
        print('Training unsupervisely')
        criterion = lambda *x: torch.mean(objective_score_edge(*x) + objective_score_node(*x))
    else:
        print('Training supervisely using the ground truth')
        criterion = get_criterion(type='crossEn')
    if args.tensorboard_savep is not None:
        writer.add_graph(model, Kp, Ke,
                         G1, G2, H1, H2, n_points)
        writer.close()
        sys.exit()
    start_epoch = CFG.TRAIN.START_EPOCH
    num_epochs = CFG.TRAIN.NUM_EPOCHS
    dataset_size = len(dataloader_train)

    # create output folders
    if not Path(CFG.TRAIN.OUTPUT_PATH).exists():
        Path(CFG.TRAIN.OUTPUT_PATH).mkdir(parents=True)

    checkpoint_path = Path(CFG.TRAIN.OUTPUT_PATH) / 'models'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    vis_path = Path(CFG.TRAIN.OUTPUT_PATH) / 'plots'
    if not vis_path.exists():
        vis_path.mkdir(parents=True)

    # load checkpoint if resuming training
    if CFG.TRAIN.START_EPOCH != 0:
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        model.load_state_dict(torch.load(model_path))

        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
        optimizer.load_state_dict(torch.load(optim_path))

    print('Start training...')

    best_acc = 0.0
    since = time.time()
    ll = []
    acc_all = []
    print(f'training is done in {device}')
    for epoch in range(start_epoch, num_epochs):

        model.train()

        epoch_loss = 0.0
        for inputs in dataloader_train:
            perm_mat = inputs['permutation_matrix'].float().to(device)
            n1_gt, n2_gt = [_.to(device) for _ in
                            [torch.Tensor([perm_mat.shape[1]]).int(), torch.Tensor([perm_mat.shape[2]]).int()]]
            n_points = [n1_gt, n2_gt]
            Kp = inputs['Kp'].float().to(device)
            Ke = inputs['Ke'].float().to(device) * args.weights_on_edges

            G1, G2 = [_ for _ in inputs['G']]
            H1, H2 = [_ for _ in inputs['H']]


            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                s_pred, _, _ = model(Kp, Ke,
                                     G1, G2, H1, H2, n_points)

                if not args.ground_truth:
                    loss = criterion(s_pred, Kp, Ke, G1.to(device), G2.to(device), H1.to(device), H2.to(device))
                else:
                    loss = criterion(s_pred, perm_mat, n1_gt, n2_gt)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * perm_mat.size(0)

        epoch_loss = epoch_loss / dataset_size
        ll.append(epoch_loss)

        print()
        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        # generate visual results in the last epoch
        acc = eval_model(model, dataloader_test, visualize=False, dst=vis_path)
        acc_all.append(acc.item())
        if acc > best_acc or epoch == num_epochs - 1:
            best_acc = acc
            torch.save(model.state_dict(), str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
            torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        scheduler.step()
        sys.stdout.flush()

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed // 60) % 60,
                                                                  time_elapsed % 60))
    sys.stdout.flush()

    return model, ll, acc_all


def resume_train_eval_model(model, optimizer, dataloader_train, dataloader_test):
    device = torch.device(args.train_device)
    # optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer)
    if not args.ground_truth:
        print('Training unsupervisely')
        criterion = lambda *x: torch.mean(objective_score_edge(*x) + objective_score_node(*x))
    else:
        print('Training supervisely using the ground truth')
        criterion = get_criterion(type='crossEn')
    if args.tensorboard_savep is not None:
        writer.add_graph(model, Kp, Ke,
                         G1, G2, H1, H2, n_points)
        writer.close()
        sys.exit()
    start_epoch = CFG.TRAIN.START_EPOCH
    num_epochs = CFG.TRAIN.NUM_EPOCHS
    dataset_size = len(dataloader_train)

    # create output folders
    if not Path(CFG.TRAIN.OUTPUT_PATH).exists():
        Path(CFG.TRAIN.OUTPUT_PATH).mkdir(parents=True)

    checkpoint_path = Path(CFG.TRAIN.OUTPUT_PATH) / 'models'
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True)

    vis_path = Path(CFG.TRAIN.OUTPUT_PATH) / 'plots'
    if not vis_path.exists():
        vis_path.mkdir(parents=True)

    # load checkpoint if resuming training
    if CFG.TRAIN.START_EPOCH != 0:
        model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
        model.load_state_dict(torch.load(model_path))

        optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
        optimizer.load_state_dict(torch.load(optim_path))

    print('Start training...')

    best_acc = 0.0
    since = time.time()
    ll = []
    acc_all = []
    print(f'training is done in {device}')
    for epoch in range(start_epoch, num_epochs):

        model.train()

        epoch_loss = 0.0
        for inputs in dataloader_train:
            perm_mat = inputs['permutation_matrix'].float().to(device)
            n1_gt, n2_gt = [_.to(device) for _ in
                            [torch.Tensor([perm_mat.shape[1]]).int(), torch.Tensor([perm_mat.shape[2]]).int()]]
            n_points = [n1_gt, n2_gt]
            Kp = inputs['Kp'].float().to(device)
            Ke = inputs['Ke'].float().to(device) * args.weights_on_edges

            G1, G2 = [_ for _ in inputs['G']]
            H1, H2 = [_ for _ in inputs['H']]


            with torch.set_grad_enabled(True):
                optimizer.zero_grad()
                s_pred, _, _ = model(Kp, Ke,
                                     G1, G2, H1, H2, n_points)

                if not args.ground_truth:
                    loss = criterion(s_pred, Kp, Ke, G1.to(device), G2.to(device), H1.to(device), H2.to(device))
                else:
                    loss = criterion(s_pred, perm_mat, n1_gt, n2_gt)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item() * perm_mat.size(0)

        epoch_loss = epoch_loss / dataset_size
        ll.append(epoch_loss)

        print()
        print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))
        # generate visual results in the last epoch
        acc = eval_model(model, dataloader_test, visualize=False, dst=vis_path)
        acc_all.append(acc.item())
        if acc > best_acc or epoch == num_epochs - 1:
            best_acc = acc
            torch.save(model.state_dict(), str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
            torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

        scheduler.step()
        sys.stdout.flush()

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed // 60) % 60,
                                                                  time_elapsed % 60))
    sys.stdout.flush()

    return model, ll, acc_all



if args.Train:
    path_train = args.exp_traind
    path_test = args.exp_evald
    print('Loading training sets...')
    # dataset_shape = SHAPEGRAPH(path=path_train, inference=True)
    dataset_shape = GMDataset(path_train)
    print('Loading testing sets...')
    train_loader = DataLoader(dataset_shape, batch_size=1, shuffle=True)
    # dataset_shape = SHAPEGRAPH(path=path_test, inference=True)
    dataset_shape = GMDataset(path_test)
    test_loader = DataLoader(dataset_shape, batch_size=1, shuffle=False)

    print(len(train_loader), len(test_loader))

    # for inputs in dataloader['train']:
    #  draw_pair(inputs)

    # quit()

    device = torch.device(args.train_device)
    model = Net()
    # model = torch.nn.DataParallel(model)
    model.to(device)
    _, ll, acc_all = train_eval_model(model, train_loader, test_loader)

if args.Resume_training:
    print('resume training from checkpoint')
    path_train = args.exp_traind
    path_test = args.exp_evald
    print('Loading training sets...')
    # dataset_shape = SHAPEGRAPH(path=path_train, inference=True)
    dataset_shape = GMDataset(path_train)
    print('Loading testing sets...')
    train_loader = DataLoader(dataset_shape, batch_size=1, shuffle=True)
    # dataset_shape = SHAPEGRAPH(path=path_test, inference=True)
    dataset_shape = GMDataset(path_test)
    test_loader = DataLoader(dataset_shape, batch_size=1, shuffle=False)

    print(len(train_loader), len(test_loader))

    # for inputs in dataloader['train']:
    #  draw_pair(inputs)

    # quit()

    device = torch.device(args.train_device)
    checkpoint = torch.load(args.checkpoint_mod)
    model = Net()
    model.to(device)
    optimizer = get_optimizer(model)
    optimizer.load_state_dict(torch.load(args.checkpoint_opt))
    model.load_state_dict(checkpoint)
    # model = torch.nn.DataParallel(model)
    _, ll, acc_all = resume_train_eval_model(model, optimizer, train_loader, test_loader)
# Inference
if args.Inference:
    # from new_code.src.dataset_PCA import GMDataset

    print('loading testing set...')
    device = torch.device(args.test_device)
    dataset_shape = GMDataset(args.exp_evald)
    test_loader = DataLoader(dataset_shape, batch_size=1, shuffle=False)

    # dataloader = {x: get_dataloader(dataset[x]) for x in ('test')}
    print('print out registrations...')
    Time = 0
    obj_edge = 0
    obj_mat_edge = 0
    obj_node = 0
    obj_mat_node = 0
    node_dis_red = 0
    edge_dis_red = 0
    node_dis_red_mat = 0
    edge_dis_red_mat = 0
    model = Net().to(device)

    model.load_state_dict(torch.load(args.exp_model))
    model.eval()
    if os.path.exists(os.path.join(args.exp_savep, args.file_name)):
        print(f'{args.file_name} exists!')
        input("Press Enter to override...")
        os.remove(os.path.join(args.exp_savep, args.file_name))

    for inputs in tqdm(test_loader):

        perm_mat = inputs['permutation_matrix'].float().to(device)
        n1_gt, n2_gt = [_.cuda() for _ in inputs['num_nodes']]
        n_points = [n1_gt, n2_gt]
        perm_mat = perm_mat[:, :n1_gt, :n2_gt]

        Kp = inputs['Kp'].float().to(device)
        Ke = inputs['Ke'].float().to(device) * args.weights_on_edges
        G1, G2 = [_ for _ in inputs['G']]
        H1, H2 = [_ for _ in inputs['H']]

        start_time = time.time()
        with torch.set_grad_enabled(False):
            s_pred, _, _ = model(Kp, Ke,
                                 G1, G2, H1, H2, n_points)

        s_pred_perm = hungarian(s_pred, n1_gt, n2_gt)
        Time += (time.time() - start_time)

        A1 = inputs['A1']
        A2 = inputs['A2']
        # A1 = torch.matmul(G1, H1.transpose(1, 2))
        # A2 = torch.matmul(G2, H2.transpose(1, 2))
        G1, H1 = adj2GH(A1[0].numpy())
        G2, H2 = adj2GH(A2[0].numpy())
        _obj = -1 * objective_score_edge(s_pred_perm, Kp, Ke, G1.to(device), G2.to(device), H1.to(device),
                                         H2.to(device))
        obj_edge += _obj
        _obj_mat = -1 * objective_score_edge(perm_mat, Kp, Ke, G1.to(device), G2.to(device), H1.to(device),
                                             H2.to(device))
        obj_mat_edge += _obj_mat

        _obj = -1 * objective_score_node(s_pred_perm, Kp, Ke, G1.to(device), G2.to(device), H1.to(device),
                                         H2.to(device))
        obj_node += _obj
        _obj_mat = -1 * objective_score_node(perm_mat, Kp, Ke, G1.to(device), G2.to(device), H1.to(device),
                                             H2.to(device))
        obj_mat_node += _obj_mat

        ##################################################### Distance Metric
        _dis = Node_distance(s_pred_perm, Kp, Ke, G1.to(device), G2.to(device), H1.to(device),
                             H2.to(device))
        node_dis_red += _dis

        _dis = Edge_distance(s_pred_perm, Kp, Ke, G1.to(device), G2.to(device), H1.to(device),
                             H2.to(device))
        edge_dis_red += _dis

        _dis = Node_distance(perm_mat, Kp, Ke, G1.to(device), G2.to(device), H1.to(device),
                             H2.to(device))
        node_dis_red_mat += _dis

        _dis = Edge_distance(perm_mat, Kp, Ke, G1.to(device), G2.to(device), H1.to(device),
                             H2.to(device))
        edge_dis_red_mat += _dis

        # print(f'objective ratio [network / matlab] is: {(obj/obj_mat).item()}')
        # print(f'objective score is: {(obj/len(dataloader.dataset)).item()}')

        # with open('/home/EG-matching/new_code/time_net.txt', 'w') as f:
        #     for data in Time:
        #         f.write(data)
        #         f.write('\n')

        perm = []
        for i in range(n1_gt.item()):
            perm.append(torch.argmax(s_pred_perm[0][i]).item() + 1)
        print('/'.join(inputs['filename'][0].split('/')[-3:]), perm,
              file=open(os.path.join(args.exp_savep, args.file_name), "a"))

    print(f'Permutation done in {Time} sec for {len(test_loader)} pairs!')
    print(f'edge objective ratio [network / matlab] is: {(obj_edge / obj_mat_edge).item()}')
    print(f'node objective ratio [network / matlab] is: {(obj_node / obj_mat_node).item()}')
    print(f'edge objective score is: {(obj_edge / len(test_loader)).item()}')
    print(f'node objective score is: {(obj_node / len(test_loader)).item()}')


