# %%
import code
import os
from numpy import float32
# os.chdir('/home/EG-matching')
import torch
import itertools
import sys
from models.BBGM.affinity_layer import InnerProductWithWeightsAffinity
from models.BBGM.sconv_archs import SiameseSConvOnNodes, SiameseNodeFeaturesToEdgeFeatures
import numpy as np
from src.feature_align import feature_align
# from src.factorize_graph_matching import construct_aff_mat, kronecker_sparse, kronecker_torch
from src.utils.pad_tensor import pad_tensor
from models.NGM.gnn import GNNLayer
from src.lap_solvers.sinkhorn import Sinkhorn, GumbelSinkhorn
from src.lap_solvers.hungarian import hungarian
# from src.sparse_torch import CSRMatrix3d
from src1.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir

cfg_from_file('experiments/ngm_qaplib.yaml')
# cfg_from_file('experiments/vgg16_ngmv2_willow.yaml')
# from src.lap_solvers.sinkhorn import Sinkhorn
from src.lap_solvers.hungarian import hungarian
from src.backbone import *
from src.evaluation_metric import objective_score
from src.utils.gpu_memory import gpu_free_memory
import math
import sys

# sys.path.append('/home/EG-matching/new_code/src')
sys.path.append('new_code/src')
from new_code.src.dataset import GMDataset, get_dataloader
from new_code.src.model import get_optimizer, get_scheduler, get_criterion, matching_accuracy
import torch.optim as optim
from Shape.Statistics import *
# dataset = {x: GMDataset(x) for x in ('train', 'test')}
# dataloader = {x: get_dataloader(dataset[x], shuffle=(x == 'train')) for x in ('train', 'test')}

# print(len(dataset['train']), len(dataset['test']))


# # %%
# for inputs in dataloader['test']:

#     data1, data2 = [_.cuda() for _ in inputs['node_embeddings']]

#     n1_gt, n2_gt = [_.cuda() for _ in [torch.Tensor([data1.shape[1]]).int(), torch.Tensor([data2.shape[1]]).int()]]

#     perm_mat = inputs['permutation_matrix'].float().cuda()
#     Kp = inputs['Kp'].float().cuda()
#     Ke = inputs['Ke'].float().cuda()
#     G1, G2 = [_ for _ in inputs['G']]
#     H1, H2 = [_ for _ in inputs['H']]


# Kro_G = [kronecker_sparse(x, y).astype(np.float32) for x, y in zip(G2,\
#      G1)]

# Kro_H = [kronecker_sparse(x, y).astype(np.float32) for x, y in zip(H2,\
#      H1)]
# Kro_G = CSRMatrix3d(Kro_G).cuda()
# Kro_H = CSRMatrix3d(Kro_H).transpose().cuda()

# #%%
# K = construct_aff_mat(Ke, Kp, Kro_G, Kro_H)

# n_points = [n1_gt, n2_gt]


# %%
from EG import Net


def EG_Mean_iteratively(EG: dict, I: int, model_path: str, sim=False):
    # path = '/home/skyler/Dropbox/Research/FDA/deep_learning_project/ThinkMatch-master/new_code/Resolution_pairs/dataset/pair1_8.mat'
    # data = loadmat(path)
    # pair1 = data['pair']
    # path1 = '/home/skyler/Dropbox/Research/FDA/deep_learning_project/ThinkMatch-master/new_code/Resolution_pairs/dataset/pair1_9.mat'
    # data = loadmat(path)
    # pair2 = data['pair']
    # I = 10
    # EG = [pair1['G1'], pair1['G2'], pair2['G1']]
    model = Net().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    n = len(EG)
    for i in range(n):
        EG[i] = {key: np.asarray(value, dtype=np.float32) for key, value in EG[i].items()}

    muEG = EG[Find_max_EG(EG)]
    EGp = []

    for iter in range(I):
        EGp = []
        print(f'starting iteration {iter}')
        for i in range(n):
            n2 = muEG['A'].shape[0]
            n1 = EG[i]['A'].shape[0]
            n1_gt = torch.from_numpy(np.asarray([n1])).cuda()
            n2_gt = torch.from_numpy(np.asarray([n2])).cuda()
            n_points = [n2_gt, n2_gt]

            tmp = EG[i]
            tmp, _ = addOneWayNullNodes(tmp, muEG)
            tmp = centeringEG(tmp)
            muEG = centeringEG(muEG)
            index_null = np.argwhere(np.sum(tmp['A'], axis=1) == 0.)
            wn = get_null_weights(tmp, muEG, 1.)
            Ke = Numpy_to_Pytorch(computeKQ(tmp['beta'], muEG['beta'])).float().cuda()
            Kp = Numpy_to_Pytorch(computeKP(tmp['nodeXY'], muEG['nodeXY'], index_null, wn)).float().cuda()
            G1, H1 = adj2GH(tmp['A'])
            G2, H2 = adj2GH(muEG['A'])
            G1 = Numpy_to_Pytorch(G1)
            G2 = Numpy_to_Pytorch(G2)
            H1 = Numpy_to_Pytorch(H1)
            H2 = Numpy_to_Pytorch(H2)
            # p = pair1['Perm_v']
            with torch.set_grad_enabled(False):
                s_pred, _ = model(Kp, Ke, Kro_G, Kro_H, n_points)

            s_pred_perm = hungarian(s_pred, n2_gt, n2_gt)
            perm = []
            for i in range(n2_gt.item()):
                perm.append(torch.argmax(s_pred_perm[0][i]).item())
            p = pDeep_to_pFqm(perm, n2, n2)
            tmp1, _ = addOneWayNullNodes(tmp, muEG)
            tmp1 = permutateElasticGraphs(tmp1, p, muEG)
            if sim == False:
                pp, qq = assignNullNodes(tmp1, muEG, p, n2 - int(index_null.shape[0]), n2)
            else:
                pp, qq = assignNullNodes(tmp1, muEG, p, n2 - int(index_null.shape[0]), n2,
                                         simu=True, index_null=index_null)
            EGp.append(pp)
            muEG = qq.copy()
        print(f'Node affinity min_max:{Kp.min().item()}, {Kp.max().item()}')
        print(f'Edge affinity min_max:{Ke.min().item()}, {Ke.max().item()}')
        if iter == I - 1:
            muEG = avgEG1(EGp, False, True)
            for i in range(n):
                a = interpEGAbeta(EGp[i].copy())
                EGp[i] = a.copy()
        else:
            muEG = avgEG1(EGp, False, True)

    return (muEG, EG, EGp)

# ww = shape2curve(tmp['beta'][:, :, 0], np.array([[1.], [2.]]), np.array([[4.], [6.]]))
