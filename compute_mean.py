# %% Mean
import numpy as np
import torch

from Shape.Mean import EG_Mean_iteratively, loadmat
import time
import sys
import os
from glob import glob
from tqdm import tqdm
from scipy.io import savemat

sys.path.append('lib_matlab')
import matlab.engine
from Shape.Statistics import *
from src1.lap_solvers.hungarian import hungarian
import argparse

parser = argparse.ArgumentParser(description="Shape graph mean computation")
parser.add_argument('--path-to-save', type=str, default='Mean_results/mean_60_PCA1', help='path to save ouput .mat file')
parser.add_argument('--data-path', type=str, default='mean_60/**', help='dataset folder to compute the mean')
parser.add_argument('--model-path', type=str, default='new_code/results1/models_PCA/real_60.pt', help='pre-trained model path')
parser.add_argument('--model-type', type=str, default='SGM', help='SGM|PCA|CIE')                    
parser.add_argument('--I', type=int, default=12, help="Number of iterations")

args, _ = parser.parse_known_args()
eng = matlab.engine.start_matlab()
eng.addpath('Matlab_functions')
# %%
model_type = args.model_type
model_path = args.model_path
data_path = args.data_path
path_to_save = args.path_to_save
if not os.path.exists(path_to_save):
    os.makedirs(path_to_save)

I = args.I

path = os.path.join(data_path, '*t')
files = glob(data_path)
print('Loading datasets to compute mean...')
EG = []
for f1 in tqdm(files):
    EG.append(loadmat(f1)['pair']['G1'])
    EG.append(loadmat(f1)['pair']['G2'])


def adj2GH(A):
    n = int(A.shape[0])
    m = 0
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0.:
                m += 1
    m = int(m)
    # m = int(np.sum(A, axis=(0, 1)))
    G = np.zeros((n, m))
    H = np.zeros_like(G)
    edge_idx = 0
    for i in range(n):
        range_j = range(i, n)
        for j in range_j:
            if A[i, j] > 0.:
                G[i, edge_idx] = 1
                G[j, edge_idx + int(m / 2)] = 1
                H[j, edge_idx] = 1
                H[i, edge_idx + int(m / 2)] = 1
                edge_idx += 1
    return (G, H)


# %%


if model_type == 'SGM':
    from EG import Net
elif model_type == 'PCA':
    from new_code.src.model import Net
elif model_type == 'CIE':
    from CIE import Net
else:
    raise 'Unknown model type'

model = Net().cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

start_time = time.time()
n = len(EG)
# n = 1
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
        n_points = [n1_gt, n2_gt]
        tmp = EG[i]
        # tmp, _ = addOneWayNullNodes(tmp, muEG)
        # if iter == 0:
        tmp = centeringEG(tmp)
        muEG = centeringEG(muEG)
        index_null = np.argwhere(np.sum(tmp['A'], axis=1) == 0.)
        wn = get_null_weights(tmp, muEG, 1.)
        if model_type == 'SGM':
            beta1 = matlab.double(tmp['beta'].tolist())
            beta2 = matlab.double(muEG['beta'].tolist())
            Ke = np.asarray(eng.computeKQ_D(beta1, beta2, False, 0.5), dtype=np.float32)
            Ke = Numpy_to_Pytorch(Ke).float().cuda()
            node1 = matlab.double(tmp['nodeXY'].tolist())
            node2 = matlab.double(muEG['nodeXY'].tolist())
            Kp = eng.computeKP_D(node1, node2, .5, matlab.double([wn]), matlab.double(index_null.tolist()), 0.)
            Kp = Numpy_to_Pytorch(np.asarray(Kp, dtype=np.float32)).float().cuda()

            # Ke = Numpy_to_Pytorch(computeKQ(tmp['beta'], muEG['beta'])).float().cuda()
            # Kp = Numpy_to_Pytorch(computeKP(tmp['nodeXY'], muEG['nodeXY'], index_null, wn)).float().cuda()
            G1, H1 = adj2GH(tmp['A'])
            G2, H2 = adj2GH(muEG['A'])
            G1 = Numpy_to_Pytorch(G1).float()
            G2 = Numpy_to_Pytorch(G2).float()
            H1 = Numpy_to_Pytorch(H1).float()
            H2 = Numpy_to_Pytorch(H2).float()
            with torch.set_grad_enabled(False):
                s_pred, _, _ = model(Kp, Ke,
                                     G1, G2, H1, H2, n_points)
            #
            s_pred_perm = hungarian(s_pred, n1_gt, n2_gt)
        elif model_type == 'PCA':
            data1 = Numpy_to_Pytorch(tmp['nodeXY'].T) / 100.
            data2 = Numpy_to_Pytorch(muEG['nodeXY'].T) / 100.
            data1 = data1 - torch.mean(data1, dim=1, keepdim=True)
            data2 = data2 - torch.mean(data2, dim=1, keepdim=True)
            G1_adjacency_matrix = tmp['A']
            G2_adjacency_matrix = muEG['A']
            G1_num_nodes = data1.shape[-2]
            G2_num_nodes = data2.shape[-2]
            G1_adjacency_matrix_with_embeddings = tmp['Abeta']
            G2_adjacency_matrix_with_embeddings = muEG['Abeta']
            G1_adjacency_list = []
            G1_edge_embeddings = []
            G2_adjacency_list = []
            G2_edge_embeddings = []
            for i in range(G1_num_nodes):
                for j in range(i + 1, G1_num_nodes):
                    if G1_adjacency_matrix[i, j] > 0:
                        for ii, jj in [[i, j], [j, i]]:
                            G1_adjacency_list.append([ii, jj])
                            # G1_edge_embeddings.append(G1_adjacency_matrix_with_embeddings[:,:,i,j].T.reshape(-1))
                            # normalize edge embeddings
                            pts = G1_adjacency_matrix_with_embeddings[:, :, ii, jj].T
                            angle = np.arctan2(pts[-1, 1] - pts[0, 1], pts[-1, 0] - pts[0, 0])
                            c, s = np.cos(angle), np.sin(angle)
                            rot = np.asarray([[c, s], [-s, c]])
                            pts = np.matmul(rot, pts.T).T
                            t = np.mean(pts[[0, -1]], axis=0)
                            scale = (pts[-1, 0] - pts[0, 0]) / 2.0
                            pts -= t
                            pts /= scale
                            G1_edge_embeddings.append(pts.flatten())

            for i in range(G2_num_nodes):
                for j in range(i + 1, G2_num_nodes):
                    if G2_adjacency_matrix[i, j] > 0:
                        for ii, jj in [[i, j], [j, i]]:
                            G2_adjacency_list.append([ii, jj])
                            # G1_edge_embeddings.append(G1_adjacency_matrix_with_embeddings[:,:,i,j].T.reshape(-1))
                            # normalize edge embeddings
                            pts = G2_adjacency_matrix_with_embeddings[:, :, ii, jj].T
                            angle = np.arctan2(pts[-1, 1] - pts[0, 1], pts[-1, 0] - pts[0, 0])
                            c, s = np.cos(angle), np.sin(angle)
                            rot = np.asarray([[c, s], [-s, c]])
                            pts = np.matmul(rot, pts.T).T
                            t = np.mean(pts[[0, -1]], axis=0)
                            scale = (pts[-1, 0] - pts[0, 0]) / 2.0
                            pts -= t
                            pts /= scale
                            G2_edge_embeddings.append(pts.flatten())
            edge1, edge2 = np.asarray(G1_edge_embeddings), np.asarray(G2_edge_embeddings)

            A1idx = np.zeros((G1_num_nodes + edge1.shape[-2], G1_num_nodes + edge1.shape[-2],),
                             np.float32)

            for i, (u, v) in enumerate(G1_adjacency_list):
                A1idx[u, v] = 1.0
                A1idx[u, G1_num_nodes + i] = 1.0
                A1idx[G1_num_nodes + i, v] = 1.0

            A2idx = np.zeros((G2_num_nodes + edge2.shape[-2], G2_num_nodes + edge2.shape[-2],),
                             np.float32)

            for i, (u, v) in enumerate(G2_adjacency_list):
                A2idx[u, v] = 1.0
                A2idx[u, G2_num_nodes + i] = 1.0
                A2idx[G2_num_nodes + i, v] = 1.0

            # data1 = torch.from_numpy(data1)
            # data2 = torch.from_numpy(data2)
            edge1 = Numpy_to_Pytorch(edge1)
            edge2 = Numpy_to_Pytorch(edge2)
            A1idx = Numpy_to_Pytorch(A1idx)
            A2idx = Numpy_to_Pytorch(A2idx)
            with torch.set_grad_enabled(False):
                s_pred, _, _ = model(data1.float().cuda(),
                                     data2.float().cuda(), edge1.float().cuda(),
                                     edge2.float().cuda(), A1idx.float().cuda(), A2idx.float().cuda())
            #
            s_pred_perm = hungarian(s_pred[:, :G1_num_nodes, :G2_num_nodes], n1_gt, n2_gt)
        elif model_type == 'CIE':
            data1 = Numpy_to_Pytorch(tmp['nodeXY'].T) / 100.
            data2 = Numpy_to_Pytorch(muEG['nodeXY'].T) / 100.
            data1 = data1 - torch.mean(data1, dim=1, keepdim=True)
            data2 = data2 - torch.mean(data2, dim=1, keepdim=True)
            G1_adjacency_matrix = tmp['A']
            G2_adjacency_matrix = muEG['A']
            G1_num_nodes = data1.shape[-2]
            G2_num_nodes = data2.shape[-2]
            G1_adjacency_matrix_with_embeddings = tmp['Abeta']
            G2_adjacency_matrix_with_embeddings = muEG['Abeta']

            # data1 = torch.from_numpy(data1)
            # data2 = torch.from_numpy(data2)
            edge1 = Numpy_to_Pytorch(G1_adjacency_matrix_with_embeddings)
            edge2 = Numpy_to_Pytorch(G2_adjacency_matrix_with_embeddings)
            edge1, edge2 = torch.flatten(edge1.permute(0, 3, 4, 1, 2), start_dim=-2, end_dim=-1), \
                           torch.flatten(edge2.permute(0, 3, 4, 1, 2), start_dim=-2, end_dim=-1)
            A1 = Numpy_to_Pytorch(G1_adjacency_matrix)
            A2 = Numpy_to_Pytorch(G2_adjacency_matrix)
            with torch.set_grad_enabled(False):
                s_pred = model(data1.float().cuda(),
                               data2.float().cuda(), edge1.float().cuda(),
                               edge2.float().cuda(), A1.float().cuda(), A2.float().cuda())
            #
            s_pred_perm = hungarian(s_pred[:, :G1_num_nodes, :G2_num_nodes], n1_gt, n2_gt)

        #     # p = pair1['Perm_v']

        perm = []
        for i in range(n1_gt.item()):
            perm.append(torch.argmax(s_pred_perm[0][i]).item())
        p = pDeep_to_pFqm(perm, n1, n2)
        tmp1, _ = addOneWayNullNodes(tmp, muEG)
        tmp1 = permutateElasticGraphs(tmp1, p, muEG)
        #     if sim == False:
        pp, qq = assignNullNodes(tmp1, muEG, p, n1, n2)
        #     else:
        #         pp, qq = assignNullNodes(tmp1, muEG, p, n2 - int(index_null.shape[0]), n2,
        #                                  simu=True, index_null=index_null)
        EGp.append(pp)
        muEG = qq.copy()
    # print(f'Node affinity min_max:{Kp.min().item()}, {Kp.max().item()}')
    # print(f'Edge affinity min_max:{Ke.min().item()}, {Ke.max().item()}')
    if iter == I - 1:
        muEG = avgEG1(EGp, False, True)
        for i in range(n):
            a = interpEGAbeta(EGp[i].copy())
            EGp[i] = a.copy()
    else:
        muEG = avgEG1(EGp, False, True)

    plt.figure()
    plotElasticGraph(muEG['Abeta'], muEG['A'], nodeLabel=False, FontSize=15, color='b', threshold=0.3)
    plt.axis('off')
    plt.savefig(os.path.join(path_to_save, 'iter' + str(iter + 1) + '.png'), dpi=300)
    savemat(os.path.join(path_to_save, 'MuEG_real_' + model_type + str(iter + 1) + '.mat'), muEG)
print(f'Finishes in:{(time.time() - start_time) // 60} min')
