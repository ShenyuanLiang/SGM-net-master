# %%
import string
# from cv2 import threshold
import numpy as np
# import cv2
import os
# import glob
import scipy.io as spio
import torch
from scipy import integrate

import optimum_reparamN2 as orN
import scipy
import matplotlib.pyplot as plt
from numba import jit, njit


# import fdasrsf as fs
# import fdasrsf.curve_functions as fcf

def optimum_reparam_curve(q1, q2, lam=0.0, method="DP"):
    """
    calculates the warping to align srsf q2 to q1
    :param q1: matrix of size nxN or array of NxM samples of first SRVF
    :param time: vector of size N describing the sample points
    :param q2: matrix of size nxN or array of NxM samples samples of second SRVF
    :param lam: controls the amount of elasticity (default = 0.0)
    :param method: method to apply optimization (default="DP") options are "DP" or "RBFGS"
    :rtype: vector
    :return gam: describing the warping function used to align q2 with q1
    """
    time = np.linspace(0, 1, q2.shape[1])
    if method == "DP":
        # gam = orN.coptimum_reparam_curve(np.ascontiguousarray(q1), time,
        #                                  np.ascontiguousarray(q2), lam)
        gam = orN.coptimum_reparam_curve(np.ascontiguousarray(q1), time,
                                         np.ascontiguousarray(q2), lam, nbhd_dim=6)

    return gam


import warnings

warnings.filterwarnings("ignore")


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''

    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list

    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def addNullNodes(EG1, n2):
    if n2 == 0:
        EG1s = EG1.copy()
    n1 = EG1['A'].shape[0]
    m = EG1['G'].shape[1]
    A1s = np.zeros([n1 + n2, n1 + n2])
    A1s[:n1, :n1] = EG1['A'].copy()
    Abetas = np.zeros([EG1['Abeta'].shape[0], EG1['Abeta'].shape[1], n1 + n2, n1 + n2])
    Abetas[:, :, :n1, :n1] = EG1['Abeta'].copy()
    Gs = np.concatenate((EG1['G'].copy(), np.zeros((n2, m))), axis=0)
    Hs = np.concatenate((EG1['H'].copy(), np.zeros((n2, m))), axis=0)
    EG1s = EG1.copy()
    EG1s['A'] = A1s
    EG1s['Abeta'] = Abetas
    EG1s['G'] = Gs
    EG1s['H'] = Hs
    EG1s['nodeXY'] = np.concatenate((EG1['nodeXY'].copy(), np.zeros((EG1['nodeXY'].shape[0], n2))), axis=1)
    return EG1s


def addOneWayNullNodes(EG1, EG2):
    n1 = EG1['A'].shape[0]
    n2 = EG2['A'].shape[0]
    if n1 <= n2:
        EG1s = addNullNodes(EG1, n2 - n1)
        EG2s = EG2.copy()
    else:
        EG2s = addNullNodes(EG2, n1 - n2)
        EG1s = EG1.copy()
    return (EG1s, EG2s)


def InnerProd_Q(q1, q2):
    dx = np.linspace(0, 1, num=q1.shape[1], endpoint=True)
    return np.trapz(np.sum(q1 * q2, 0), dx)


def L2Dist(q1, q2):
    dx = np.linspace(0, 1, num=q1.shape[1], endpoint=True)
    val = np.sqrt(np.trapz(np.sum((np.abs(q1 - q2)) ** 2, 0), dx))
    return val


def curve_to_q(p, scale=False):
    T = p.shape[1];
    d = p.shape[0]
    v = np.zeros((d, T))
    for i in range(d):
        v[i, :] = np.gradient(p[i, :], 1 / (T - 1))
    L = np.zeros((T,))
    q = np.zeros((d, T))
    for i in range(T):
        L[i] = np.sqrt(np.linalg.norm(v[:, i]))
        if L[i] > .000001:
            q[:, i] = v[:, i] / L[i]
        else:
            q[:, i] = 0. * np.ones((d,))
    if scale:
        q = q / (np.sqrt(InnerProd_Q(q, q)) + np.spacing(1))
    return q


def q_to_curve(q, scale=1):
    n, T = q.shape
    qnorm = np.zeros(T)
    for i in range(0, T):
        qnorm[i] = np.linalg.norm(q[:, i])

    beta = np.zeros((n, T))
    for i in range(0, n):
        beta[i, :] = integrate.cumtrapz(q[i, :] * qnorm, initial=0) / (T - 1)

    beta = scale * beta

    return (beta)


def group_action_by_gamma_coord(f, gamma):
    n, T = f.shape
    fn = np.zeros((n, T))

    for j in range(0, n):
        s = scipy.interpolate.interp1d(np.linspace(0, 1, T), f[j, :])
        fn[j, :] = s(gamma)

    return (fn)


def Group_Action_by_Gamma_Coord(X, gam):
    T = X.shape[1]
    idx = np.around(gam * T) - 1
    for i in range(len(idx)):
        if idx[i] <= 0:
            idx[i] = 0
    for i in range(len(idx)):
        if idx[i] >= T - 1:
            idx[i] = T - 1
    idx = idx.astype(int)
    # idx = np.min(np.max(0, idx), T)
    Xn = X[:, idx]
    return Xn


def Find_Best_Rotation(q1, q2, identity=False):
    d, _ = q1.shape
    if identity == True:
        R = np.eye(d)
        q2new = R @ q2
        return q2new, R
    else:
        A = q1 @ q2.T
        U, _, V = np.linalg.svd(A)
        if np.linalg.det(A) > 0:
            S = np.eye(d)
        else:
            S = np.eye(d)
            S[:, -1] = -S[:, -1]

        R = U @ S @ V
        q2new = R @ q2
        return (q2new, R)


def Find_Rotation_and_Seed_unique(q1, q2):
    minE = -999999
    q2best = q2
    for ctr in range(1):
        q2n, R = Find_Best_Rotation(q1, q2best, identity=True)
        if np.linalg.norm(q1 - q2n, 'fro') > .0001:
            gamI = optimum_reparam_curve(q1, q2n, 0.0)
            # xx = matlab.double(q1.tolist())
            # yy = matlab.double(q2n.tolist())
            # gamI = np.asarray(eng.DynamicProgrammingQ_Adam(yy, xx, 0.0, 0.0))
            # gamI = gamI.reshape((gamI.shape[1], ))
            gamI = (gamI - gamI[0]) / (gamI[-1] - gamI[0])
            p2n = q_to_curve(q2n)
            p2new = Group_Action_by_Gamma_Coord(p2n, gamI)
            # p2new = group_action_by_gamma_coord(p2n, gamI)
            q2new = curve_to_q(p2new)
        else:
            q2new = q2n

        Ec = InnerProd_Q(q1, q2new)
        if Ec > minE:
            Rbest = R
            q2best = q2new
            minE = Ec
    #  else:
    #      Rbest = R
    #      q2best = q2new
    #      minE = Ec
    return (q2best, Rbest)


# def Find_Rotation_and_Seed_unique(q1,q2):
#     [q2_best, Rbest, _] = fcf.find_rotation_and_seed_unique(q1, q2, rotation=False)

#     return (q2_best, Rbest)

def ElasticShootingVector(q1, q2):
    q2n, _ = Find_Rotation_and_Seed_unique(q1, q2)
    d = (InnerProd_Q(q1, q2n))
    return (d, q2n)


def get_null_weights(EG1, EG2, percent=1.):
    n1 = EG1['A'].shape[0]
    n2 = EG2['A'].shape[0]
    row_index2 = np.argwhere(np.sum(EG1['A'], axis=1) == 0.)
    D = np.zeros([n2, n2])
    for i in range(n1):
        for j in range(n2):
            if np.in1d(i, row_index2):
                D[i, j] = 0.
            else:
                nodeXY1 = EG1['nodeXY'][:, i]
                nodeXY2 = EG2['nodeXY'][:, j]
                D[i, j] = np.linalg.norm(nodeXY1 - nodeXY2)
    d_node = np.trace(D)
    w = d_node / n2 * percent
    return w


# from fdasrsf.curve_functions import elastic_shooting_vector
def computeKQ(beta1, beta2, sigma=.5):
    r = beta1.shape[-1]
    c = beta2.shape[-1]
    KQ = np.zeros((2 * r, 2 * c))
    d_edge = []
    for i in range(r):
        for j in range(c):
            q1 = curve_to_q(beta1[:, :, i])
            q2 = curve_to_q(beta2[:, :, j])
            tmp = L2Dist(q1, q2) + 1 * \
                  np.abs(np.sqrt(InnerProd_Q(q1, q1)) -
                         np.sqrt(InnerProd_Q(q2, q2)))
            d_edge.append(tmp)

    c1 = c + 1;
    c2 = 2 * c
    for i in range(r):
        for j in range(c1 - 1, c2):
            q1 = curve_to_q(beta1[:, :, i])
            q2 = curve_to_q(np.fliplr(beta2[:, :, j - c]))
            tmp = L2Dist(q1, q2) + 1 * \
                  np.abs(np.sqrt(InnerProd_Q(q1, q1)) -
                         np.sqrt(InnerProd_Q(q2, q2)))
            d_edge.append(tmp)

    dMax = np.max(np.asarray(d_edge))

    for i in range(r):
        for j in range(c):
            q1 = curve_to_q(beta1[:, :, i])
            q2 = curve_to_q(beta2[:, :, j])
            dd = L2Dist(q1, q2) + 1 * \
                 np.abs(np.sqrt(InnerProd_Q(q1, q1)) -
                        np.sqrt(InnerProd_Q(q2, q2)))
            # _,tmp, _ = elastic_shooting_vector(q1, q2)
            tmp = 1 - dd / dMax
            KQ[i, j] = tmp

    for i in range(r, 2 * r):
        for j in range(c, 2 * c):
            KQ[i, j] = KQ[i - r, j - c]

    c1 = c + 1;
    c2 = 2 * c
    for i in range(r):
        for j in range(c1 - 1, c2):
            q1 = curve_to_q(beta1[:, :, i])
            q2 = curve_to_q(np.fliplr(beta2[:, :, j - c]))
            dd = L2Dist(q1, q2) + 1 * \
                 np.abs(np.sqrt(InnerProd_Q(q1, q1)) -
                        np.sqrt(InnerProd_Q(q2, q2)))
            # _,tmp, _ = elastic_shooting_vector(q1, q2)
            tmp = 1 - dd / dMax
            KQ[i, j] = tmp

    for i in range(r, 2 * r):
        for j in range(c):
            KQ[i, j] = KQ[i - r, j + c]

    # minKQ = np.min(np.min(KQ))
    # if minKQ <0:
    #     KQ = KQ + 2 * np.abs(minKQ)
    KQ = sigma * KQ

    return KQ


def computeKP(nodeXY1, nodeXY2, row_index2, wd, sigma=.5):
    r = nodeXY1.shape[1]
    c = nodeXY2.shape[1]
    KP = np.zeros((r, c))
    D = np.zeros([r, r])
    for i in range(r):
        for j in range(c):
            if np.in1d(i, row_index2):
                D[i, j] = wd
            else:
                D[i, j] = np.linalg.norm(nodeXY1[:, i] - nodeXY2[:, j])
    dMax = np.max(D)

    for i in range(r):
        for j in range(c):
            if np.in1d(i, row_index2):
                KP[i, j] = 1 - wd / dMax
            else:
                KP[i, j] = 1 - np.linalg.norm(nodeXY1[:, i] - nodeXY2[:, j]) / dMax
    KP = sigma * KP
    return KP


def Numpy_to_Pytorch(object):
    a = torch.from_numpy(object)
    a = a.unsqueeze(0)
    return a


def pDeep_to_pFqm(perm_deep, n1, n2):
    perm_deep = np.asarray(perm_deep, dtype=np.float32)
    b = np.arange(n2)
    label = np.setdiff1d(b, perm_deep)
    num = len(label)
    perm_deep2 = np.zeros(n2)
    perm_deep2[:n1] = perm_deep
    j = 0
    for i in range(n1, n2):
        perm_deep2[i] = label[j]
        j += 1
    perm_deep3 = np.zeros(n2)
    for i in range(n2):
        perm_deep3[i] = np.argwhere(perm_deep2 == i)
    return perm_deep3.astype(int)


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


def adj2beta(Abeta, A):
    n = A.shape[0]
    m = 0
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0.:
                m += 1
    m = int(m)
    k = 0
    beta = np.zeros((Abeta.shape[0], Abeta.shape[1], int(m / 2)))
    for i in range(n):
        for j in range(i + 1, n):
            if A[i, j] > 0.:
                beta[:, :, k] = Abeta[:, :, i, j]
                k += 1
    return beta


def permutateElasticGraphs(EG1, p, EG2):
    Abeta1p = EG1['Abeta'][:, :, p, :][:, :, :, p].copy()
    A1p = EG1['A'][p, :][:, p].copy()
    n = A1p.shape[0]
    print('registering edges')
    A2 = EG2['A'].copy()
    Abeta2 = EG2['Abeta'].copy()
    for i in range(n):
        for j in range(i + 1, n):
            if A1p[i, j] > 0.0 and A2[i, j] > 0.0:
                q1 = curve_to_q(Abeta1p[:, :, i, j])
                q2 = curve_to_q(Abeta2[:, :, i, j])
                Abeta1p[:, :, i, j] = q_to_curve(Find_Rotation_and_Seed_unique(q2, q1)[0])
    nodeXY1 = np.zeros_like(EG1['nodeXY'])
    # assert np.isnan(np.sum(Abeta1p)) == True, 'Nan values'
    for i in range(nodeXY1.shape[1]):
        nodeXY1[:, i] = EG1['nodeXY'][:, p[i]].copy()
    G, H = adj2GH(A1p)
    beta = adj2beta(Abeta1p, A1p)
    EG1p = EG1.copy()
    EG1p['Abeta'] = Abeta1p.copy()
    EG1p['nodeXY'] = nodeXY1.copy()
    EG1p['A'] = A1p.copy()
    EG1p['beta'] = beta.copy()
    EG1p['G'] = G.copy()
    EG1p['H'] = H.copy()
    return EG1p


def assignNullNodes(EG1p, EG2, p, n1, n2, simu=False, index_null=None):
    assert n1 <= n2, 'n1 should be less or equal to n2'
    if simu == False:
        for i in range(n2):
            if p[i] >= n1:
                EG1p['nodeXY'][:, i] = EG2['nodeXY'][:, i].copy()
    else:
        for i in range(n2):
            if np.in1d(p[i], index_null):
                EG1p['nodeXY'][:, i] = EG2['nodeXY'][:, i].copy()
    return (EG1p, EG2)


def shape2curve(rawShape, start, ending):
    v1 = -rawShape[:, -1] + rawShape[:, 0]
    v2 = -ending + start
    v2 = v2.reshape((v2.shape[0]))
    if np.linalg.norm(v2) == 0.:
        curve = np.zeros(rawShape.shape)
    else:

        qq = np.sum(v1.conj() * v2, axis=0) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        theta = np.arccos(round(qq, 4))

        if np.imag(theta) != 0.:
            theta = np.real(theta)
        c = np.cos(theta)
        s = -np.sin(theta)
        o1 = np.asarray([[c, s], [-s, c]])
        theta = -theta
        c = np.cos(theta)
        s = -np.sin(theta)
        o2 = np.asarray([[c, s], [-s, c]])
        if np.linalg.norm(o1 @ v1 - v2) < np.linalg.norm(o2 @ v1 - v2):
            rawShape = o1 @ rawShape
        else:
            rawShape = o2 @ rawShape

        s = np.linalg.norm(ending - start) / np.linalg.norm(rawShape[:, 0] - \
                                                            rawShape[:, -1])
        rawShape = rawShape * s
        a = (start.reshape((start.shape[0],)) - rawShape[:, 0])
        curve = rawShape + a.reshape([a.shape[0], 1])

        assert np.isnan(np.sum(curve)) == False, 'Nan values'

    return curve


def adjbeta2q(Abeta, A, rescale=False):
    Abetaq = np.zeros_like(Abeta)
    n = A.shape[0]
    l = 0
    for i in range(n):
        for j in range(n):
            if A[i, j] > 0.:
                q = curve_to_q(Abeta[:, :, i, j], rescale)
                l = l + np.sqrt(InnerProd_Q(q, q))
                Abetaq[:, :, i, j] = q
    l = l / 2
    return (Abetaq, l)


def adjq2beta(Abetaq, A):
    n = A.shape[0]
    Abeta = np.zeros_like(Abetaq)
    for i in range(n):
        for j in range(i + 1, n):
            Abeta[:, :, i, j] = q_to_curve(Abetaq[:, :, i, j])
            Abeta[:, :, j, i] = np.fliplr(Abeta[:, :, i, j])
    return Abeta


def interpEGAbeta(EG):
    n = EG['A'].shape[0]
    nodeXY = EG['nodeXY']
    for i in range(n):
        for j in range(i + 1, n):
            if EG['A'][i, j] > 0.:
                EG['Abeta'][:, :, i, j] = shape2curve(EG['Abeta'][:, :, i, j], \
                                                      nodeXY[:, i], nodeXY[:, j])
                EG['Abeta'][:, :, j, i] = np.fliplr(EG['Abeta'][:, :, i, j])
    return EG


def avgEG1(EG, rescale, interp):
    n = len(EG)
    nnode = EG[0]['A'].shape[0]
    d = EG[0]['Abeta'].shape[0]
    T = EG[0]['Abeta'].shape[1]
    muA = np.zeros((nnode, nnode))
    muAbetaq = np.zeros((d, T, nnode, nnode))
    muNodeXY = np.zeros((d, nnode))
    for i in range(n):
        muA = muA + EG[i]['A']
        tmp, l = adjbeta2q(EG[i]['Abeta'], EG[i]['A'])
        if rescale:
            tmp = tmp / l
        muAbetaq = muAbetaq + tmp
        muNodeXY = muNodeXY + EG[i]['nodeXY']
    ##################
    muA = muA / n
    # muA[muA <= .15] = 0.
    # muA[muA > .3] = 1.
    ##################
    muAbeta = adjq2beta(muAbetaq / n, muA)
    muNodeXY = muNodeXY / n
    G, H = adj2GH(muA)
    if interp:
        n_mu = muA.shape[0]
        for i in range(n_mu):
            for j in range(i + 1, n_mu):
                if muA[i, j] > 0.:
                    muAbeta[:, :, i, j] = shape2curve(muAbeta[:, :, i, j], \
                                                      muNodeXY[:, i], muNodeXY[:, j])
    muEG = dict({'beta': adj2beta(muAbeta, muA), 'A': muA, 'G': G, \
                 'H': H, 'Abeta': muAbeta, 'nodeXY': muNodeXY})
    return muEG


def Find_max_EG(EG):
    a = np.zeros([len(EG), ])
    for i in range(len(EG)):
        A = EG[i]['A']
        index_null = np.argwhere(np.sum(A, axis=1) == 0.)
        a[i] = A.shape[0] - index_null.shape[0]
    idx = np.argmax(a)
    return int(idx)


def plotElasticGraph(Abeta, A, nodeLabel=False, FontSize=16, color='b', threshold=0.):
    node_color = 'r'
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[1]):
            if A[i, j] == 1.:
                plt.plot(Abeta[0, :, i, j], Abeta[1, :, i, j], color, linewidth=4)
                if nodeLabel:
                    plt.plot(Abeta[0, 0, i, j], Abeta[1, 0, i, j], color=node_color,
                             marker='o', linewidth=1)
                    plt.text(Abeta[0, 0, i, j], Abeta[1, 0, i, j], str(i + 1), fontsize=FontSize)
                    plt.plot(Abeta[0, -1, i, j], Abeta[1, -1, i, j], color=node_color,
                             marker='o', linewidth=1)
                    plt.text(Abeta[0, -1, i, j], Abeta[1, -1, i, j], str(j + 1), fontsize=FontSize)
            elif A[i, j] < 1. and A[i, j] > threshold:
                plt.plot(Abeta[0, :, i, j], Abeta[1, :, i, j], color=color, linewidth=4 * A[i, j])
                if nodeLabel:
                    plt.plot(Abeta[0, 0, i, j], Abeta[1, 0, i, j], color=node_color,
                             marker='o', linewidth=1)
                    plt.text(Abeta[0, 0, i, j], Abeta[1, 0, i, j], str(i + 1), fontsize=FontSize)
                    plt.plot(Abeta[0, -1, i, j], Abeta[1, -1, i, j], color=node_color,
                             marker='o', linewidth=1)
                    plt.text(Abeta[0, -1, i, j], Abeta[1, -1, i, j], str(j + 1), fontsize=FontSize)


def computeElasticGeodesic(beta1, beta2, alpha, linearFlag):
    q1 = curve_to_q(beta1)
    q2 = curve_to_q(beta2)
    if linearFlag:
        qa = alpha * q1 + (1 - alpha) * q2
    else:
        q1n, _ = Find_Rotation_and_Seed_unique(q2, q1)
        qa = alpha * q1n + (1 - alpha) * q2
    c = q_to_curve(qa)
    return c


def computeElasticGraphGeodesic(EG1, EG2, a, Aa, nodeXYa, linearFlag):
    Abeta1 = EG1['Abeta']
    Abeta2 = EG2['Abeta']
    Abeta_a = np.zeros_like(Abeta1)
    m1 = Aa.shape[0]
    m2 = m1
    nullEdge = np.zeros((Abeta1.shape[0], Abeta1.shape[1]))
    if a == 1.:
        Abeta_a = Abeta1
    elif a == 0.:
        Abeta_a = Abeta2
    else:
        for i in range(m1):
            for j in range(i + 1, m2):
                if Aa[i, j] == 1.:
                    rawShape = computeElasticGeodesic(Abeta1[:, :, i, j], Abeta2[:, :, i, j],
                                                      a, linearFlag)
                    Abeta_a[:, :, i, j] = shape2curve(rawShape, nodeXYa[:, i], nodeXYa[:, j])
                elif Aa[i, j] == 0.:
                    continue
                else:
                    if EG1['A'][i, j] == 1.:
                        rawShape = computeElasticGeodesic(Abeta1[:, :, i, j],
                                                          nullEdge, a, linearFlag)
                        Abeta_a[:, :, i, j] = shape2curve(rawShape, nodeXYa[:, i],
                                                          nodeXYa[:, j])
                    else:
                        rawShape = computeElasticGeodesic(nullEdge, Abeta2[:, :, i, j],
                                                          a, linearFlag)
                        Abeta_a[:, :, i, j] = shape2curve(rawShape, nodeXYa[:, i],
                                                          nodeXYa[:, j])
    return Abeta_a


def plotEGcombination_label(EG1, EG2, a, nodeLabel, translation=0., threshold=0.):
    A1 = EG1['A']
    A2 = EG2['A']
    dim = EG1['Abeta'].shape[0]
    aA = a * A1 + (1 - a) * A2
    nodeXYa = a * EG1['nodeXY'] + (1 - a) * EG2['nodeXY'] + translation
    aAbeta = computeElasticGraphGeodesic(EG1, EG2, a, aA, nodeXYa, True)
    if dim == 2:
        plotElasticGraph(aAbeta, aA, nodeLabel, 15, threshold=threshold)
    else:
        print('Error: undeveloped for plot3ElasticGraph')
    plt.title('a = ' + str(round(a, 1)), fontsize=20)


def computeEGDistance(EG1, EG2, Lambda=1., scale=True):
    A1, A2 = EG1['A'], EG2['A']
    Abeta1, Abeta2 = EG1['Abeta'], EG2['Abeta']
    Abetaq1 = np.zeros_like(Abeta1)
    Abetaq2 = np.zeros_like(Abeta2)
    n = int(Abeta1.shape[2])
    l1, l2 = 0., 0.
    for i in range(n):
        for j in range(i + 1, n):
            if A1[i, j] or A2[i, j]:
                q = curve_to_q(Abeta1[:, :, i, j], False)
                l1 = l1 + np.sqrt(InnerProd_Q(q, q))
                Abetaq1[:, :, i, j] = q
                q = curve_to_q(Abeta2[:, :, i, j], False)
                l2 = l2 + np.sqrt(InnerProd_Q(q, q))
                Abetaq2[:, :, i, j] = q
    if not scale:
        l1 = 1.
        l2 = 1.
    d = 0
    for i in range(n):
        for j in range(i + 1, n):
            if A1[i, j] or A2[i, j]:
                q1 = Abetaq1[:, :, i, j] / l1
                q2 = Abetaq2[:, :, i, j] / l2
                d = d + np.linalg.norm(q1 - q2) ** 2

    d = np.sqrt(d)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(EG1['nodeXY'][:, i] - EG2['nodeXY'][:, j])
    d_node = np.trace(D)
    d = d + Lambda * d_node
    return d, d_node


def NumpyEG(EG):
    return {key: np.asarray(value, dtype=np.float32) for key, value in EG.items()}


def plot_geodesics(EG1p_deep, EG2, path_to_save: str, threshold=0.):
    a = np.linspace(1, 0, 6)
    plt.figure(figsize=[15, 10], dpi=300)

    for i in range(len(a)):
        plt.subplot(2, 3, i + 1)
        plotEGcombination_label(EG1p_deep, EG2, a[i], True, threshold=threshold)
        plt.axis('off')
        plt.axis('equal')

    # plt.show()
    plt.savefig(path_to_save, dpi=300)


def centeringEG(EG):
    nodeXY = EG['nodeXY']
    meanXY = np.mean(nodeXY, axis=1)
    EGc = EG.copy()
    for i in range(nodeXY.shape[1]):
        EGc['nodeXY'][:, i] = EGc['nodeXY'][:, i] - meanXY
    Abeta_new = np.zeros_like(EGc['Abeta'])
    for i in range(EGc['A'].shape[0]):
        for j in range(EGc['A'].shape[0]):
            if EGc['A'][i, j] > 0.:
                Abeta_new[:, :, i, j] = shape2curve(EGc['Abeta'][:, :, i, j], EGc['nodeXY'][:, i],
                                                    EGc['nodeXY'][:, j])

    EGc['Abeta'] = Abeta_new.copy()
    beta = adj2beta(Abeta_new, EGc['A'])
    EGc['beta'] = beta.copy()
    return EGc

# plt.figure()
# plotElasticGraph(EG1['Abeta'],EG1['A'],True)
# plt.axis('off')
# plt.show()
