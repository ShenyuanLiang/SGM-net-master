import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.optimize as opt
import numpy as np

from config_PCA import cfg

"""
Intra-graph convolution operation, with single convolutional layer
"""


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


def pairwise_euclidean_distance(x, y):
    """Compute the pairwise Euclidean similarity between x and y.
    This function computes the following similarity value between each pair of x_i
    and y_j: s(x_i, y_j) = -|x_i - y_j|^2.
    Args:
      x: NxD float tensor.
      y: MxD float tensor.
    Returns:
      s: NxM float tensor, the pairwise euclidean similarity.
    """

    return torch.cdist(x, y, p=2)


class Gconv(nn.Module):
    def __init__(self, in_features, out_features):
        super(Gconv, self).__init__()
        self.num_inputs = in_features
        self.num_outputs = out_features
        self.a_fc = nn.Linear(self.num_inputs, self.num_outputs)
        self.u_fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, A, x, norm=True):
        if norm is True:
            A = F.normalize(A, p=1, dim=-2)

        ax = self.a_fc(x)
        ux = self.u_fc(x)
        x = torch.bmm(A, F.relu(ax)) + F.relu(ux)  # has size (bs, N, num_outputs)

        return x


"""
Perform graph convolution on two input graphs (g1, g2)
"""


class Siamese_Gconv(nn.Module):
    def __init__(self, in_features, num_features):
        super(Siamese_Gconv, self).__init__()
        self.gconv = Gconv(in_features, num_features)

    def forward(self, g1, g2):
        emb1 = self.gconv(*g1)
        emb2 = self.gconv(*g2)
        return F.normalize(emb1, dim=-1, p=2), F.normalize(emb2, dim=-1, p=2)


"""
Computes the affinity matrix from feature space.
"""


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


"""
Computes a new row-stochastic matrix with softmax. A large number (alpha) is multiplied to the input matrix to scale up the difference.
"""


class Norm(nn.Module):
    def __init__(self, alpha=200):
        super(Norm, self).__init__()
        self.alpha = alpha
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, s):
        ret_s = torch.zeros_like(s)
        for b in range(len(s)):
            ret_s[b] = self.softmax(self.alpha * s[b])
        return ret_s


"""
Transforms an input matrix into a bi-stochastic matrix.
"""


class Sinkhorn(nn.Module):
    def __init__(self, max_iter=10, epsilon=1e-4):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.epsilon = epsilon

    def forward(self, s, n1, n2, nrows=None, ncols=None, dtype=torch.float32):
        batch_size = s.shape[0]

        row_norm_ones = torch.zeros(batch_size, s.shape[1], s.shape[1], device=s.device)  # size: row x row
        col_norm_ones = torch.zeros(batch_size, s.shape[2], s.shape[2], device=s.device)  # size: col x col
        for b in range(batch_size):
            row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
            col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
            row_norm_ones[b, row_slice, row_slice] = 1
            col_norm_ones[b, col_slice, col_slice] = 1

        # for Sinkhorn stacked on last dimension
        if len(s.shape) == 4:
            row_norm_ones = row_norm_ones.unsqueeze(-1)
            col_norm_ones = col_norm_ones.unsqueeze(-1)

        s += self.epsilon
        s[:, :n1, n2:].fill_(0)
        s[:, n1:, :n2].fill_(0)

        for i in range(self.max_iter):
            if i % 2 == 1:
                # column norm
                sum = torch.sum(torch.mul(s.unsqueeze(3), col_norm_ones.unsqueeze(1)), dim=2)
            else:
                # row norm
                sum = torch.sum(torch.mul(row_norm_ones.unsqueeze(3), s.unsqueeze(1)), dim=2)

            tmp = torch.zeros_like(s)
            for b in range(batch_size):
                row_slice = slice(0, nrows[b] if nrows is not None else s.shape[2])
                col_slice = slice(0, ncols[b] if ncols is not None else s.shape[1])
                tmp[b, row_slice, col_slice] = 1 / sum[b, row_slice, col_slice]
            s = s * tmp

        return s


"""
PCA-GM
"""


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # embeddings
        self.fc1_nodes = nn.Linear(cfg.DATASET.NODE_EMBEDDING_SIZE, cfg.MODEL.FC_HIDDEN_SIZE)
        self.fc2_nodes = nn.Linear(cfg.MODEL.FC_HIDDEN_SIZE, cfg.MODEL.GNN_FEAT_SIZE)
        self.fc1_edges = nn.Linear(cfg.DATASET.EDGE_EMBEDDING_SIZE, cfg.MODEL.FC_HIDDEN_SIZE)
        self.fc2_edges = nn.Linear(cfg.MODEL.FC_HIDDEN_SIZE, cfg.MODEL.GNN_FEAT_SIZE)

        # network
        self.node_aff = Affinity(cfg.MODEL.GNN_FEAT_SIZE)
        self.edge_aff = Affinity(cfg.MODEL.GNN_FEAT_SIZE)
        self.bi_stochastic = Sinkhorn(max_iter=cfg.MODEL.BS_ITER_NUM, epsilon=cfg.MODEL.BS_EPSILON)
        self.norm_layer = Norm(alpha=cfg.MODEL.NORM_ALPHA)
        self.gnn_layer = cfg.MODEL.GNN_LAYER
        for i in range(self.gnn_layer):
            gnn_layer = Siamese_Gconv(cfg.MODEL.GNN_FEAT_SIZE, cfg.MODEL.GNN_FEAT_SIZE)
            self.add_module('gnn_layer_{}'.format(i), gnn_layer)
            self.add_module('affinity_{}'.format(i), Affinity(cfg.MODEL.GNN_FEAT_SIZE))
            if i == self.gnn_layer - 2:
                self.add_module('cross_graph_{}'.format(i),
                                nn.Linear(cfg.MODEL.GNN_FEAT_SIZE * 2, cfg.MODEL.GNN_FEAT_SIZE))

    def forward(self, emb1, emb2, edge_emb1, edge_emb2, Aidx_src, Aidx_tgt):
        emb1 = F.relu(self.fc1_nodes(emb1[0]))
        emb1 = torch.unsqueeze(F.normalize(self.fc2_nodes(emb1), dim=-1, p=2), dim=0)

        emb2 = F.relu(self.fc1_nodes(emb2[0]))
        emb2 = torch.unsqueeze(F.normalize(self.fc2_nodes(emb2), dim=-1, p=2), dim=0)
        Kp = pairwise_euclidean_similarity(emb1, emb2)
        edge_emb1 = F.relu(self.fc1_edges(edge_emb1[0]))
        edge_emb1 = torch.unsqueeze(torch.sigmoid(self.fc2_edges(edge_emb1)), dim=0)

        edge_emb2 = F.relu(self.fc1_edges(edge_emb2[0]))
        edge_emb2 = torch.unsqueeze(torch.sigmoid(self.fc2_edges(edge_emb2)), dim=0)
        Ke = pairwise_euclidean_similarity(edge_emb1, edge_emb2)
        n1 = emb1.size()[1]
        n2 = emb2.size()[1]
        e1 = edge_emb1.size()[1]
        e2 = edge_emb2.size()[1]
        # print(n1, n2, e1, e2)

        # print(emb1.size(), edge_emb1.size(), Aidx_src.size())
        emb1 = torch.cat((emb1, edge_emb1), 1)
        emb2 = torch.cat((emb2, edge_emb2), 1)

        A_src = Aidx_src  # torch.squeeze(edge_emb1[Aidx_src], dim=-1)
        A_tgt = Aidx_tgt  # torch.squeeze(edge_emb2[Aidx_tgt], dim=-1)

        for i in range(self.gnn_layer):
            gnn_layer = getattr(self, 'gnn_layer_{}'.format(i))
            emb1, emb2 = gnn_layer([A_src, emb1], [A_tgt, emb2])
            affinity = getattr(self, 'affinity_{}'.format(i))
            s = torch.zeros(1, n1 + e1, n2 + e2).cuda()
            s[:, :n1, :n2] = affinity(emb1[:, :n1, :], emb2[:, :n2, :])
            s[:, n1:, n2:] = affinity(emb1[:, n1:, :], emb2[:, n2:, :])
            # s = affinity(emb1, emb2)
            s = self.norm_layer(s)

            s = self.bi_stochastic(s, n1, n2, [emb1.size()[1]], [emb2.size()[1]])

            if i == self.gnn_layer - 2:
                cross_graph = getattr(self, 'cross_graph_{}'.format(i))
                emb1_new = cross_graph(torch.cat((emb1, torch.bmm(s, emb2)), dim=-1))
                emb2_new = cross_graph(torch.cat((emb2, torch.bmm(s.transpose(1, 2), emb1)), dim=-1))
                emb1 = emb1_new
                emb2 = emb2_new

        return s, Kp, Ke


"""
Cross entropy loss between two permutations.
"""


class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns):
        batch_num = pred_perm.shape[0]
        pred_perm = pred_perm.to(dtype=torch.float32)

        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(pred_perm[b, :pred_ns[b], :gt_ns[b]], gt_perm[b, :pred_ns[b], :gt_ns[b]],
                                           reduction='sum')
            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)

        return loss / n_sum


class DistanceLoss(nn.Module):
    def __init__(self):
        super(DistanceLoss, self).__init__()

    def forward(self, pred_perm, data1, data2, edge1, edge2):
        batch_num = pred_perm.shape[0]
        pred_perm = pred_perm.to(dtype=torch.float32)
        n1, n2 = data1.shape[1], data2.shape[1]
        e1, e2 = edge1.shape[1], edge2.shape[1]

        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        # assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            pred_perm_node = pred_perm[b, :n1, :n2]
            pred_perm_edge = pred_perm[b, n1:, n2:]
            pred_perm_null = pred_perm[b, n1:, :n2]
            pred_perm_null_ = pred_perm[b, :n1, n2:]
            loss_node = pairwise_euclidean_distance(torch.matmul(pred_perm_node, data2),
                                                    data1)

            loss_edge = pairwise_euclidean_distance(torch.matmul(pred_perm_edge, edge2),
                                                    edge1)
            loss_null = F.binary_cross_entropy(pred_perm_null,
                                               torch.zeros_like(pred_perm_null.to(pred_perm.device)), reduction='sum')
            loss_null_ = F.binary_cross_entropy(pred_perm_null_,
                                                torch.zeros_like(pred_perm_null_.to(pred_perm.device)), reduction='sum')

            loss += torch.trace(loss_node[b, :, :]).sum() + torch.triu(loss_edge[b, :, :]).sum().sqrt() + \
                    loss_null + loss_null_

            n_sum += n1 + e1

        return loss / n_sum


class NullLoss(nn.Module):
    def __init__(self):
        super(NullLoss, self).__init__()

    def forward(self, pred_perm, data1, data2, edge1, edge2):
        batch_num = pred_perm.shape[0]
        pred_perm = pred_perm.to(dtype=torch.float32)
        n1, n2 = data1.shape[1], data2.shape[1]
        e1, e2 = edge1.shape[1], edge2.shape[1]

        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        # assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            pred_perm_null = pred_perm[b, n1:, :n2]
            pred_perm_null_ = pred_perm[b, :n1, n2:]
            # pred_perm_edge = pred_perm[b, n1:, n2:]
            # loss_edge = pairwise_euclidean_distance(torch.matmul(pred_perm_edge, edge2),
            #                                         edge1)
            loss_null = F.binary_cross_entropy(pred_perm_null,
                                               torch.zeros_like(pred_perm_null.to(pred_perm.device)), reduction='sum')
            loss_null_ = F.binary_cross_entropy(pred_perm_null_,
                                                torch.zeros_like(pred_perm_null_.to(pred_perm.device)), reduction='sum')

            loss += loss_null + loss_null_

            n_sum += n1 + e1

        return loss / n_sum


"""
Training tools.
"""


def get_optimizer(model):
    if cfg.TRAIN.OPTIMIZER == 'Adam':
        return optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'SGD':
        return optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, nesterov=True)
    else:
        return None


def get_scheduler(optimizer):
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_DECAY,
                                          last_epoch=cfg.TRAIN.START_EPOCH - 1)


def get_criterion(type='crossEn'):
    if type == 'crossEn':
        return CrossEntropyLoss()
    elif type == 'distance':
        return DistanceLoss()
    else:
        return NullLoss()


"""
Solve optimal LAP permutation by hungarian algorithm.
"""


def hungarian(s: torch.Tensor, n1=None, n2=None):
    device = s.device
    batch_num = s.shape[0]

    perm_mat = s.cpu().detach().numpy() * -1
    for b in range(batch_num):
        n1b = perm_mat.shape[1] if n1 is None else n1[b]
        n2b = perm_mat.shape[2] if n2 is None else n2[b]
        row, col = opt.linear_sum_assignment(perm_mat[b, :n1b, :n2b])
        perm_mat[b] = np.zeros_like(perm_mat[b])
        perm_mat[b, row, col] = 1
    perm_mat = torch.from_numpy(perm_mat).to(device)

    return perm_mat


"""
Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.
"""


def matching_accuracy(pmat_pred, pmat_gt, ns):
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can only contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should only contain 0/1 elements.'
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    match_num = 0
    total_num = 0
    for b in range(batch_num):
        match_num += torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]])
        total_num += torch.sum(pmat_gt[b, :ns[b]])

    return match_num / total_num, match_num, total_num
