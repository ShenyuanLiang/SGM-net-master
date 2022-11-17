#%%
import os
import sys
import cv2
import time
import torch
import numpy as np
import torch.nn.functional as f
sys.path.append('/home/skyler/Dropbox/Research/FDA/deep_learning_project/new-code/src_editable')
from dataset import GMDataset, get_dataloader
from model import Net, get_optimizer, get_scheduler, get_criterion, hungarian, matching_accuracy
from pathlib import Path
from config import cfg
# from Hungarian_torch import hungarian_torch
# sys.path.append('/home/skyler/Dropbox/Research/FDA/deep_learning_project/lib')
#
#
# import matlab.engine
# eng = matlab.engine.start_matlab()
# eng.addpath('/home/skyler/Dropbox/Research/ElasticGraphs_Editable/code/lib/ElasticShapeCode/',
# nargout=0)

def horn_absolute_orientation(pts1, pts2):
  assert pts1.shape == pts2.shape, "Both arrays must have the same dimension!"

  # get centroids (first equation in Section 2.B from [1])
  avg1 = np.mean(pts1, axis=0, keepdims=True)
  avg2 = np.mean(pts2, axis=0, keepdims=True)

  # get scale factor (ninth equation in Section 2.C from [1])
  scale = np.sqrt(np.sum(np.square(pts1-avg1))/np.sum(np.square(pts2-avg2)))

  # compute orthonormal rotation matrix (fourth equation in Section 3.A and fifth equation in Section 3.E from [1])
  M = np.matmul((pts1-avg1).T, pts2-avg2)
  R = M.dot(scipy.linalg.inv(scipy.linalg.sqrtm(M.T.dot(M))))

  # compute translation (eighth equation of Section 2.B from [1])
  t = avg1 - scale*R.dot(avg2.T).T

  # return "rigid" transformation
  return scale, R, t
#%%

# dataset = {x: GMDataset(x) for x in ('train', 'test')}
# dataloader = {x: get_dataloader(dataset[x], shuffle=(x == 'train')) for x in ('train', 'test')}
# print(len(dataset['train']), len(dataset['test']))
#
# for inputs in dataloader['train']:
#     data1, data2 = [_.cuda() for _ in inputs['node_embeddings']]
#     edge1, edge2 = [_.cuda() for _ in inputs['edge_embeddings']]
#     A1idx, A2idx = [_.cuda() for _ in inputs['adjacency_matrices']]
#     n1_gt, n2_gt = [_.cuda() for _ in inputs['num_nodes']]
#     e1_gt, e2_gt = [_.cuda() for _ in inputs['num_edges']]
#     Shape1, Shape2 = [_.cuda() for _ in inputs['Shapes']]
#     adj1, adj2 = [_.cuda() for _ in inputs['adjacency_list']]
#     Abetaq1, Abetaq2 = [_.cuda() for _ in inputs['SRVF']]
#     l1, l2 = [_.cuda() for _ in inputs['Total_dis']]
#     perm_mat = inputs['permutation_matrix'].float().cuda()

#%%
def InnerProd_Q(q1, q2):
    dx = np.linspace(0, 1, num=q1.shape[1], endpoint=True)
    return np.trapz(np.sum(q1*q2, 0), dx)

def curve_to_q(p, scale = False):
    T = p.shape[1]; d = p.shape[0]
    v = np.zeros((d, T))
    for i in range(d):
        v[i, :] = np.gradient(p[i, :], 1/(T-1))
    L = np.zeros((T, ))
    q = np.zeros((d, T))
    for i in range(T):
        L[i] = np.sqrt(np.linalg.norm(v[:, i]))
        if L[i] > .000001:
            q[:, i] = v[:, i] / L[i]
        else:
            q[:, i] = 0 * np.ones((d,))
    return q



#%%

class dis_faq:
    def __init__(self, data1, data2, n1_gt, n2_gt, Shape1, Shape2, e1_gt, e2_gt,
                 permutation, perm = True, Euclidean = False):
        self.node1 = data1[0]
        self.node2 = data2[0]
        self.n1 = n1_gt
        self.n2 = n2_gt
        self.e1 = e1_gt
        self.e2 = e2_gt
        self.edge1 = Shape1[0]
        self.edge2 = Shape2[0]
        self.perm = perm
        self.s_pred = permutation
        self.Eu = Euclidean

    def trans_perm(self):
        s_new = self.s_pred.clone().detach()
        ss = hungarian(s_new, self.n1+self.e1, self.n2+self.e2)
        ss = ss[0, :self.n1.item(), :self.n2.item()].unsqueeze(0)
        # if self.flag:
        #     ss = hungarian(s_new, self.n1, self.n2)
        # else:
        #     ss = s_new
        #ss = s_new
        t1 = np.arange(self.n2.item())
        t2 = torch.max(ss, 2).indices.cpu().numpy().reshape(-1)
        #print(t2)
        idx = np.setdiff1d(t1, t2)
        row = np.zeros([len(idx), ss.shape[2]], dtype=np.float32)
        for i, j in enumerate(idx):
            row[i, j] = 1.0
        row = torch.from_numpy(row).cuda()

        return torch.cat((self.s_pred[0, :self.n1.item(), :self.n2.item()], row), 0)

    def Edge_dis(self):

        if not self.Eu:

            if self.perm:
                Abeta1 = np.zeros((self.edge2.cpu().numpy().shape[0], self.edge2.cpu().numpy().shape[1],
                                   self.edge2.cpu().numpy().shape[2], self.edge2.cpu().numpy().shape[3]))
                Abeta2 = self.edge2.cpu().numpy()
                Abeta1[:, :, :self.n1, :self.n1] = self.edge1.cpu().numpy()
                Abetaq1 = np.zeros_like(Abeta1)
                Abetaq2 = np.zeros_like(Abeta1)
                l1 = 0; l2 = 0
                for i in range(self.n2):
                    for j in range(i + 1, self.n2):
                        if (Abeta1[:, :, i, j] != 0).any() or (Abeta2[:, :, i, j] != 0).any():
                            xx = Abeta1[:, :, i, j]
                            q = curve_to_q(xx)
                            Abetaq1[:, :, i, j] = np.array(q)
                            l1 = l1 + np.sqrt((InnerProd_Q(q, q)))
                            xx = Abeta2[:, :, i, j]
                            q = curve_to_q(xx)
                            Abetaq2[:, :, i, j] = np.array(q)
                            l2 = l2 + np.sqrt((InnerProd_Q(q, q)))

                # l1 = torch.from_numpy(l1).cuda()
                # l2 = torch.from_numpy(l2).cuda()
                Abetaq1 = torch.from_numpy(Abetaq1).double().cuda()
                Abetaq2 = torch.from_numpy(Abetaq2).double().cuda()
                s_pred_new = self.trans_perm().double()
                d = torch.linalg.norm((s_pred_new @ Abetaq2 @ s_pred_new.T) / l2 \
                                      - Abetaq1 / l1)
            else:
                Abeta1 = np.zeros((self.edge2.cpu().numpy().shape[0], self.edge2.cpu().numpy().shape[1],
                                   self.edge2.cpu().numpy().shape[2], self.edge2.cpu().numpy().shape[3]))
                Abeta2 = self.edge2.cpu().numpy()
                Abeta1[:, :, :self.n1, :self.n1] = self.edge1.cpu().numpy()
                Abetaq1 = np.zeros_like(Abeta1)
                Abetaq2 = np.zeros_like(Abeta1)
                l1 = 0
                l2 = 0
                for i in range(self.n2):
                    for j in range(i + 1, self.n2):
                        if (Abeta1[:, :, i, j] != 0).any() or (Abeta2[:, :, i, j] != 0).any():
                            xx = Abeta1[:, :, i, j]
                            q = curve_to_q(xx)
                            Abetaq1[:, :, i, j] = np.array(q)
                            l1 = l1 + np.sqrt((InnerProd_Q(q, q)))
                            xx = Abeta2[:, :, i, j]
                            q = curve_to_q(xx)
                            Abetaq2[:, :, i, j] = np.array(q)
                            l2 = l2 + np.sqrt((InnerProd_Q(q, q)))

                # l1 = torch.from_numpy(l1).cuda()
                # l2 = torch.from_numpy(l2).cuda()
                Abetaq1 = torch.from_numpy(Abetaq1).double().cuda()
                Abetaq2 = torch.from_numpy(Abetaq2).double().cuda()
                d = torch.linalg.norm(Abetaq2 / l2 - Abetaq1 / l1)
        else:
            if self.perm:
                Abeta1 = torch.zeros_like(self.edge2).cuda()
                Abeta2 = self.edge2
                Abeta1[:, :, :self.n1, :self.n1] = self.edge1
                Abetaq1 = Abeta1.double()
                Abetaq2 = Abeta2.double()
                l1 = torch.sum(torch.matmul(Abetaq1, Abetaq1)).sqrt()
                l2 = torch.sum(torch.matmul(Abetaq2, Abetaq2)).sqrt()
                s_pred_new = self.trans_perm().double()
                d = torch.linalg.norm((s_pred_new @ Abetaq2 @ s_pred_new.T) / l2 \
                                      - Abetaq1 / l1)
            else:
                Abeta1 = torch.zeros_like(self.edge2).cuda()
                Abeta2 = self.edge2
                Abeta1[:, :, :self.n1, :self.n1] = self.edge1
                Abetaq1 = Abeta1.double()
                Abetaq2 = Abeta2.double()
                l1 = torch.sum(torch.matmul(Abetaq1, Abetaq1)).sqrt()
                l2 = torch.sum(torch.matmul(Abetaq2, Abetaq2)).sqrt()
                d = torch.linalg.norm(Abetaq2 / l2 - Abetaq1 / l1)
        return d

    def Node_dis(self):

        if self.perm:
            s_pred_new = self.trans_perm().double()
            D = torch.zeros(self.n2.item(), self.n2.item()).cuda()
            node1 = torch.zeros_like(self.node2).cuda()
            node1[:self.n1.item(), :] = self.node1
            node2 = s_pred_new @ self.node2.double()
            for i in range(self.n2.item()):
                for j in range(self.n2.item()):
                    D[i, j] = torch.linalg.norm(node1[i, :] - node2[j, :])

        else:
            D = torch.zeros(self.n2.item(), self.n2.item()).cuda()
            node1 = torch.zeros_like(self.node2).cuda()
            node1[:self.n1.item(), :] = self.node1
            for i in range(self.n2.item()):
                for j in range(self.n2.item()):
                    D[i, j] = torch.linalg.norm(node1[i, :] - self.node2[j, :], 2)

        return torch.diagonal((D))

    """
    def Edge_dis_Euclidean(self):

        if self.perm:
            Abeta1 = np.zeros((self.edge2.cpu().numpy().shape[0], self.edge2.cpu().numpy().shape[1],
                               self.edge2.cpu().numpy().shape[2], self.edge2.cpu().numpy().shape[3]))
            Abeta2 = self.edge2.cpu().numpy()
            Abeta1[:, :, :self.n1, :self.n1] = self.edge1.cpu().numpy()
            Abetaq1 = np.zeros_like(Abeta1)
            Abetaq2 = np.zeros_like(Abeta1)
            l1 = 0; l2 = 0
            for i in range(self.n2):
                for j in range(i + 1, self.n2):
                    if (Abeta1[:, :, i, j] != 0).any() or (Abeta2[:, :, i, j] != 0).any():
                        xx = Abeta1[:, :, i, j]
                        q = xx
                        Abetaq1[:, :, i, j] = np.array(q)
                        l1 = l1 + np.sqrt(np.array((np.inner(q, q))))
                        xx = Abeta2[:, :, i, j]
                        q = xx
                        Abetaq2[:, :, i, j] = np.array(q)
                        l2 = l2 + np.sqrt(np.array((np.inner(q, q))))

            # l1 = torch.from_numpy(l1).cuda()
            # l2 = torch.from_numpy(l2).cuda()
            Abetaq1 = torch.from_numpy(Abetaq1).double().cuda()
            Abetaq2 = torch.from_numpy(Abetaq2).double().cuda()
            s_pred_new = self.trans_perm().double()
            d = torch.linalg.norm((s_pred_new @ Abetaq2 @ s_pred_new.T) / l2 \
                                  - Abetaq1 / l1)
        else:
            Abeta1 = np.zeros((self.edge2.cpu().numpy().shape[0], self.edge2.cpu().numpy().shape[1],
                               self.edge2.cpu().numpy().shape[2], self.edge2.cpu().numpy().shape[3]))
            Abeta2 = self.edge2.cpu().numpy()
            Abeta1[:, :, :self.n1, :self.n1] = self.edge1.cpu().numpy()
            Abetaq1 = np.zeros_like(Abeta1)
            Abetaq2 = np.zeros_like(Abeta1)
            l1 = 0
            l2 = 0
            for i in range(self.n2):
                for j in range(i + 1, self.n2):
                    if (Abeta1[:, :, i, j] != 0).any() or (Abeta2[:, :, i, j] != 0).any():
                        xx = matlab.double(Abeta1[:, :, i, j].tolist())
                        q = eng.curve_to_q(xx, False)
                        Abetaq1[:, :, i, j] = np.array(q)
                        l1 = l1 + np.sqrt(np.array((eng.InnerProd_Q(q, q))))
                        xx = matlab.double(Abeta2[:, :, i, j].tolist())
                        q = eng.curve_to_q(xx, False)
                        Abetaq2[:, :, i, j] = np.array(q)
                        l2 = l2 + np.sqrt(np.array((eng.InnerProd_Q(q, q))))

            # l1 = torch.from_numpy(l1).cuda()
            # l2 = torch.from_numpy(l2).cuda()
            Abetaq1 = torch.from_numpy(Abetaq1).double().cuda()
            Abetaq2 = torch.from_numpy(Abetaq2).double().cuda()
            d = torch.linalg.norm(Abetaq2 / l2 - Abetaq1 / l1)
        return d
    """
    def dis(self):

        node = self.Node_dis()
        #weights = 10 ** (-str(torch.sum(node).item()).find('.'))
        weights = 1
        edge = self.Edge_dis()
        return edge*weights + node



#%%
class dis_faq1:
    def __init__(self, data1, data2, n1_gt, n2_gt, e1_gt, e2_gt,
                 permutation, l1, l2, Abetaq1, Abetaq2, perm = True, Euclidean = False):
        self.node1 = data1[0]
        self.node2 = data2[0]
        self.n1 = n1_gt
        self.n2 = n2_gt
        self.e1 = e1_gt
        self.e2 = e2_gt
        self.q1 = Abetaq1[0]
        self.q2 = Abetaq2[0]
        self.l1 = l1
        self.l2 = l2
        self.perm = perm
        self.s_pred = permutation
        self.Eu = Euclidean

    def trans_perm(self):
        s_new = self.s_pred.clone().detach()
        ss = hungarian(s_new, self.n1+self.e1, self.n2+self.e2)
        ss = ss[0, :self.n1.item(), :self.n2.item()].unsqueeze(0)
        # if self.flag:
        #     ss = hungarian(s_new, self.n1, self.n2)
        # else:
        #     ss = s_new
        #ss = s_new
        t1 = np.arange(self.n2.item())
        t2 = torch.max(ss, 2).indices.cpu().numpy().reshape(-1)
        #print(t2)
        idx = np.setdiff1d(t1, t2)
        row = np.zeros([len(idx), ss.shape[2]], dtype=np.float32)
        for i, j in enumerate(idx):
            row[i, j] = 1.0
        row = torch.from_numpy(row).cuda()
        # s_pred_new = torch.cat((ss[0, :self.n1.item(), :self.n2.item()], row), 0).double()
        # A2 = s_pred_new @ self.A2[0].double() @ s_pred_new.T
        # s_hun = hungarian_torch(self.s_pred[:, :self.n1.item(), :self.n2.item()], self.n1, self.n2)
        # return torch.cat((s_hun[0], row), 0)
        results = torch.cat((self.s_pred[0, :self.n1.item(), :self.n2.item()], row), 0)
        return results.double()

    def Edge_dis(self):

        if not self.Eu:

            if self.perm:

                Abetaq1 = self.q1.double()
                Abetaq2 = self.q2.double()
                #Abetaq1[:, :, :self.n1, :self.n1] =self.q1.double()
                s_pred_new = self.trans_perm()
                Abetaq2p = s_pred_new @ Abetaq2 @ s_pred_new.T
                d = torch.linalg.norm(Abetaq2p / self.l2 \
                                      - Abetaq1 / self.l1)
            # if self.perm:
            #
            #     Abetaq1 = torch.zeros_like(self.q2).double()
            #     Abetaq2 = self.q2.double()
            #     Abetaq1[:, :, :self.n1, :self.n1] =self.q1.double()
            #     s_pred_new = self.trans_perm().double()
            #     a1 = s_pred_new @ Abetaq2 @ s_pred_new.T
            #     a2 = Abetaq1
            #     d = 0
            #     for i in range(self.n2):
            #         for j in range(i + 1, self.n2):
            #             if (a1[:, :, i, j] != 0).any() or (a2[:, :, i, j] != 0).any():
            #                 xx = a1[:, :, i, j] / self.l1
            #                 yy = a2[:, :, i, j] / self.l2
            #                 d = d + torch.linalg.norm(xx - yy) ** 2
            #     d = d.sqrt()

            else:

                Abetaq1 = self.q1.double()
                Abetaq2 = self.q2.double()

                d = torch.linalg.norm(Abetaq2 / self.l2 - Abetaq1 / self.l1)
        else:
            if self.perm:
                Abeta1 = torch.zeros_like(self.edge2).cuda()
                Abeta2 = self.edge2
                Abeta1[:, :, :self.n1, :self.n1] = self.edge1
                Abetaq1 = Abeta1.double()
                Abetaq2 = Abeta2.double()
                l1 = torch.sum(torch.matmul(Abetaq1, Abetaq1)).sqrt()
                l2 = torch.sum(torch.matmul(Abetaq2, Abetaq2)).sqrt()
                s_pred_new, A2 = self.trans_perm()
                d = torch.linalg.norm((s_pred_new @ Abetaq2 @ s_pred_new.T) / self.l2 \
                                      - Abetaq1 / self.l1)
            else:
                Abeta1 = torch.zeros_like(self.edge2).cuda()
                Abeta2 = self.edge2
                Abeta1[:, :, :self.n1, :self.n1] = self.edge1
                Abetaq1 = Abeta1.double()
                Abetaq2 = Abeta2.double()
                l1 = torch.sum(torch.matmul(Abetaq1, Abetaq1)).sqrt()
                l2 = torch.sum(torch.matmul(Abetaq2, Abetaq2)).sqrt()
                d = torch.linalg.norm(Abetaq2 / self.l2 - Abetaq1 / self.l1)
        return d.sqrt()

    def Node_dis(self):

        if self.perm:
            s_pred_new = self.trans_perm()
            D = torch.zeros(self.n2.item(), self.n2.item()).cuda()
            node1 = torch.zeros_like(self.node2).cuda()
            node1[:self.n1.item(), :] = self.node1
            node2 = s_pred_new @ self.node2.double()
            for i in range(self.n2.item()):
                for j in range(self.n2.item()):
                    D[i, j] = torch.linalg.norm(node1[i, :] - node2[j, :])

        else:
            D = torch.zeros(self.n2.item(), self.n2.item()).cuda()
            node1 = torch.zeros_like(self.node2).cuda()
            node1[:self.n1.item(), :] = self.node1
            for i in range(self.n2.item()):
                for j in range(self.n2.item()):
                    D[i, j] = torch.linalg.norm(node1[i, :] - self.node2[j, :])

        return torch.diagonal(D)

    def dis(self):

        node = self.Node_dis()
        #weights = 10 ** (-str(torch.sum(node).item()).find('.'))
        weights = 100
        edge = self.Edge_dis()
        return weights*edge + node

#%%


# sd = dis_faq1(data1, data2, n1_gt, n2_gt,e1_gt, e2_gt,
#                   perm_mat, l1, l2, Abetaq1, Abetaq2, perm = True)
#
# print(f'Original edge distance:{sd.Edge_dis().sum()}')
# print(f'Original node distance:{sd.Node_dis().sum()}')
# print(f'Original distance:{sd.dis().sum()}')


#%%
"""

#%%
sd = dis_faq(data1, data2, n1_gt, n2_gt, Shape1, Shape2, e1_gt, e2_gt,
                  perm_mat, perm = True)

print(f'gt edge distance:{sd.Edge_dis_Euclidean()}')


#%%
# import time
# start_time = time.time()
# sd = dis_faq(data1, data2, n1_gt, n2_gt, Shape1, Shape2, e1_gt, e2_gt,
#                   perm_mat, perm = False)
# 
# dis_org = sd.dis()
# 
# sd = dis_faq(data1, data2, n1_gt, n2_gt, Shape1, Shape2, e1_gt, e2_gt,
#                   perm_mat, perm = True)
# 
# dis_perm = sd.dis()
# #%%
# f.l1_loss((dis_perm), (dis_org), reduction='sum')
# print("--- %s seconds ---" % (time.time() - start_time))
# #%%
# dis_perm = torch.FloatTensor(dis_perm)
# dis_org = torch.FloatTensor(dis_org)
# 
# F.l1_loss((dis_perm), (dis_org), reduction='mean')
# #%%
# 
# A1idx_num = A1idx.cpu().numpy()[0]
# 
# edge_indices = {(u,v):i for i, (u,v) in enumerate(adj1[0, :,:].cpu().numpy())}
Original edge distance:434.78378473810557
"""
#%%

# model = Net().cuda()
# model.load_state_dict(torch.load(os.path.join(cfg.TRAIN.OUTPUT_PATH,'models','params_0045.pt')))
# with torch.set_grad_enabled(False):
#     s_pred = model(data1, data2, edge1, edge2, A1idx, A2idx)
# s_pred_perm = hungarian(s_pred, n1_gt + e1_gt, n2_gt + e2_gt)
# sd = dis_faq1(data1, data2, n1_gt, n2_gt,e1_gt, e2_gt,
#                   s_pred_perm, l1, l2, Abetaq1, Abetaq2, perm = False)
# print(f'Original edge distance:{sd.Edge_dis().sum()}')
# print(f'Original node distance:{sd.Node_dis().sum()}')
#
#
# sd = dis_faq1(data1, data2, n1_gt, n2_gt,e1_gt, e2_gt,
#                   perm_mat, l1, l2, Abetaq1, Abetaq2, perm = True)
# print(f'Original edge distance:{sd.Edge_dis().sum()}')
# print(f'Original node distance:{sd.Node_dis().sum().item()}')


#%%
# Dis = [];Dis_org = []; Dis_edge = []; Dis_node = []; Dis_edge_org = []; Dis_node_org = []
# DDis = [];DDis_org = []; DDis_edge = []; DDis_node = []; DDis_edge_org = []; DDis_node_org = []
# with torch.set_grad_enabled(False):
#     s_pred = model(data1, data2, edge1, edge2, A1idx, A2idx)
# s_pred_perm = hungarian(s_pred, n1_gt+e1_gt, n2_gt+e2_gt)
# sd = dis_faq1(data1, data2, n1_gt, n2_gt, e1_gt, e2_gt,
#             s_pred_perm, l1, l2, Abetaq1, Abetaq2, perm=True)
#
# sdo = dis_faq1(data1, data2, n1_gt, n2_gt,e1_gt, e2_gt,
#              s_pred_perm, l1, l2, Abetaq1, Abetaq2, perm = False)
#
# dis = sd.Node_dis().sum().item() * 0.001 + sd.Edge_dis().item()
# dis_org = sdo.Node_dis().sum().item() * 0.001 + sdo.Edge_dis().item()
# Dis.append(dis); Dis_org.append(dis_org)
# Dis_edge.append(sd.Edge_dis().item()); Dis_node.append(sd.Node_dis().sum().item())
# Dis_edge_org.append(sdo.Edge_dis().item()); Dis_node_org.append(sdo.Node_dis().sum().item())
#
# sd = dis_faq1(data1, data2, n1_gt, n2_gt, e1_gt, e2_gt,
#           perm_mat, l1, l2, Abetaq1, Abetaq2, perm=True)
#
# sdo = dis_faq1(data1, data2, n1_gt, n2_gt,e1_gt, e2_gt,
#              perm_mat, l1, l2, Abetaq1, Abetaq2, perm = False)
#
# dis = sd.Node_dis().sum().item() * 0.001 + sd.Edge_dis().item()
# dis_org = sdo.Node_dis().sum().item() * 0.001 + sdo.Edge_dis().item()
# DDis.append(dis); DDis_org.append(dis_org)
# DDis_edge.append(sd.Edge_dis().item()); DDis_node.append(sd.Node_dis().sum().item())
# DDis_edge_org.append(sdo.Edge_dis().item()); DDis_node_org.append(sdo.Node_dis().sum().item())