import os
import torch
import random
import scipy.io
import numpy as np
from tqdm import tqdm
from config import cfg


class GraphPairDataset:
    def __init__(self, path):
        # assert mode == 'train' or mode == 'test', 'No match found for dataset {}'.format(mode)
        # self.mode = mode
        self.path = path
        self.pairs = self.__load_pairs()

    # load MATLAB file
    @staticmethod
    def __load_MATLAB(filename):
        graph_pair = scipy.io.loadmat(filename)

        # get 1st graph
        G1_node_embeddings = np.asarray(graph_pair['pair']['G1'][0][0]['nodeXY'][0][0], dtype=np.float32).T
        G1_num_nodes, node_embedding_size = G1_node_embeddings.shape
        # assert node_embedding_size == cfg.DATASET.NODE_EMBEDDING_SIZE

        G1_adjacency_matrix = np.asarray(graph_pair['pair']['G1'][0][0]['A'][0][0])
        G1_adjacency_matrix_with_embeddings = np.asarray(graph_pair['pair']['G1'][0][0]['Abeta'][0][0])
        # G1_adjacency_matrix_with_embeddings = np.asarray(graph_pair['pair']['Abetaq11'][0][0])
        G_edgeaff = np.asarray(graph_pair['pair']['Kq'][0][0])
        G_nodeaff = np.asarray(graph_pair['pair']['Kp'][0][0])
        G1_H = np.asarray(graph_pair['pair']['G1'][0][0]['H'][0][0])
        G1_G = np.asarray(graph_pair['pair']['G1'][0][0]['G'][0][0])
        G1_adjacency_list = []
        G1_edge_embeddings = []
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
                        # t = np.mean(pts[[0,-1]], axis=0)
                        scale = (pts[-1, 0] - pts[0, 0]) / 2.0
                        # pts -= t
                        pts /= scale
                        G1_edge_embeddings.append(pts.flatten())

        # get 2nd graph and find null nodes
        G2_node_embeddings = np.asarray(graph_pair['pair']['G2'][0][0]['nodeXY'][0][0], dtype=np.float32).T
        G2_num_nodes, node_embedding_size = G2_node_embeddings.shape
        # assert node_embedding_size == cfg.DATASET.NODE_EMBEDDING_SIZE
        # assert G1_num_nodes == G2_num_nodes
        G2_H = np.asarray(graph_pair['pair']['G2'][0][0]['H'][0][0])
        G2_G = np.asarray(graph_pair['pair']['G2'][0][0]['G'][0][0])
        G2_adjacency_matrix = np.asarray(graph_pair['pair']['G2'][0][0]['A'][0][0])

        G2_adjacency_matrix_with_embeddings = np.asarray(graph_pair['pair']['G2'][0][0]['Abeta'][0][0])
        # G2_adjacency_matrix_with_embeddings = np.asarray(graph_pair['pair']['Abetaq22'][0][0])
        G2_adjacency_list = []
        G2_edge_embeddings = []
        ###valid_nodes = [False for _ in range(G2_num_nodes)]
        for i in range(G2_num_nodes):
            for j in range(i + 1, G2_num_nodes):
                if G2_adjacency_matrix[i, j] > 0:
                    for ii, jj in [[i, j], [j, i]]:
                        G2_adjacency_list.append([ii, jj])
                        ###valid_nodes[ii] = True
                        ###valid_nodes[jj] = True
                        # G2_edge_embeddings.append(G2_adjacency_matrix_with_embeddings[:,:,i,j].T.reshape(-1))
                        # normalize edge embeddings
                        pts = G2_adjacency_matrix_with_embeddings[:, :, ii, jj].T
                        angle = np.arctan2(pts[-1, 1] - pts[0, 1], pts[-1, 0] - pts[0, 0])
                        c, s = np.cos(angle), np.sin(angle)
                        rot = np.asarray([[c, s], [-s, c]])
                        pts = np.matmul(rot, pts.T).T
                        # t = np.mean(pts[[0,-1]], axis=0)
                        # pts -= t
                        scale = (pts[-1, 0] - pts[0, 0]) / 2.0
                        pts /= scale
                        G2_edge_embeddings.append(pts.flatten())

        # get permutation vector and delete null nodes
        # print(G1_num_nodes, G2_num_nodes, graph_pair['pair']['Perm_v'][0][0][0].shape, np.amin(graph_pair['pair']['Perm_v'][0][0][0]), np.amax(graph_pair['pair']['Perm_v'][0][0][0]))
        # permutation_vector_ = np.asarray(graph_pair['pair']['Perm_v'][0][0][0])-1
        # permutation_vector = permutation_vector_[valid_nodes]
        permutation_vector = np.asarray(graph_pair['pair']['Perm_v'][0][0][0]) - 1

        # avg1 = np.mean(G1_node_embeddings, axis=0, keepdims=True)
        # G1_node_embeddings -= avg1;
        # avg2 = np.mean(G2_node_embeddings, axis=0, keepdims=True)
        # G2_node_embeddings -= avg2;
        if 'bottom' in filename:
            G1_node_embeddings[:, 1] *= -1
            G2_node_embeddings[:, 1] *= -1

        # convert lists to ndarrays
        # G1_adjacency_list = np.asarray(G1_adjacency_list, dtype=np.int64)
        G1_edge_embeddings = np.asarray(G1_edge_embeddings, dtype=np.float32)
        # G1_num_edges, edge_embedding_size = G1_edge_embeddings.shape
        # assert edge_embedding_size == cfg.DATASET.EDGE_EMBEDDING_SIZE
        #
        # G2_adjacency_list = np.asarray(G2_adjacency_list, dtype=np.int64)
        G2_edge_embeddings = np.asarray(G2_edge_embeddings, dtype=np.float32)
        # G2_num_edges, edge_embedding_size = G2_edge_embeddings.shape
        # assert edge_embedding_size == cfg.DATASET.EDGE_EMBEDDING_SIZE

        # return smaller graph (always 2nd graph in MATLAB) as the first graph
        # return permutation_vector, \
        #         {'num_nodes': G1_num_nodes, 'edge_embeddings': G1_edge_embeddings, 'node_embeddings': G1_node_embeddings, 'G_edgeaff':G_edgeaff, 'G_nodeaff': G_nodeaff, 'G':G1_G, 'H':G1_H}, \
        #         {'num_nodes': G2_num_nodes, 'edge_embeddings': G2_edge_embeddings,  'node_embeddings': G2_node_embeddings, 'G':G2_G, 'H':G2_H}
        return permutation_vector, \
               {'num_nodes': G1_num_nodes, 'node_embeddings': G1_node_embeddings, 'G_edgeaff': G_edgeaff,
                'G_nodeaff': G_nodeaff, 'G': G1_G, 'H': G1_H, 'A': G1_adjacency_matrix,
                'edge_embeddings': G1_edge_embeddings}, \
               {'num_nodes': G2_num_nodes, 'node_embeddings': G2_node_embeddings, 'G': G2_G, 'H': G2_H,
                'A': G2_adjacency_matrix, 'edge_embeddings': G2_edge_embeddings}

    # load graph file
    @staticmethod
    def load_pair(filename):
        # assert cfg.DATASET.TYPE == 'MATLAB', 'Dataset type "{}" is invalid, check configuration file'.format(
        #     cfg.DATASET.TYPE)

        # load MATLAB file with a pair of graphs
        permutation_vector, G1, G2 = GraphPairDataset.__load_MATLAB(filename)

        # return graph pair
        return {'filename': filename, 'permutation_vector': permutation_vector, 'G1': G1, 'G2': G2}

    # preload all dataset pairs
    def __load_pairs(self):
        pairs = []

        # create repeatable dataset split
        files = []
        for root, ds, fs in os.walk(self.path):
            for f in fs:
                if f.endswith('.mat'):
                    files.append(os.path.join(root, f))
        files = sorted(files)

        # files = sorted(os.listdir(cfg.DATASET.ROOT_DIR))
        # random.seed(cfg.DATASET.SPLIT_SEED)
        # random.shuffle(files)

        for k, filename in enumerate(tqdm(files)):
            # if k < cfg.DATASET.SPLIT_INDEX and self.mode == 'train' or k >= cfg.DATASET.SPLIT_INDEX and self.mode == 'test':
            pair = self.load_pair(filename)
            pairs.append(pair)
        # if len(pairs) >= 10:
        #  break

        return pairs

    # number of graph pairs
    def __len__(self):
        return len(self.pairs)

    # return one pair of graphs
    def get_pair(self, idx):
        assert idx >= 0 and idx < len(self.pairs), 'Pair with index {} does not exist'.format(idx)
        return self.pairs[idx]


class GMDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.ds = GraphPairDataset(path)
        self.path = path

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pair = self.ds.get_pair(idx)

        # compute adjacency matrix for each graph
        # adjacency_matrices = []
        # for G in [pair['G1'], pair['G2']]:
        #   adjacency_matrix = np.zeros((G['num_nodes'], G['num_nodes']), np.float32)
        #
        #   for i, (u,v) in enumerate(G['adjacency_list']):
        #     adjacency_matrix[u, v] = 1.0
        #     adjacency_matrix[u, G['num_nodes']+i] = 1.0
        #     adjacency_matrix[G['num_nodes']+i, v] = 1.0
        #
        #   adjacency_matrices.append(torch.from_numpy(adjacency_matrix))

        # compute the permutation matrix between the two graphs
        permutation_matrix = np.zeros((pair['G1']['num_nodes'], pair['G2']['num_nodes']), np.int32)
        for i, j in enumerate(pair['permutation_vector']):
            permutation_matrix[i, j] = 1

        # apply random rigid transformation to G1
        pts = pair['G1']['node_embeddings'].copy()
        # if self.mode == 'train':
        #   angle = np.random.uniform(-1, 1)*(np.pi/18.0)
        #   c, s = np.cos(angle), np.sin(angle)
        #   rot = np.asarray([[c,s],[-s,c]])
        #   pts = np.matmul(rot,pts.T).T
        #   t = np.random.normal(0,0.1,size=(1,2))
        #   pts += t
        # apply random flip
        """
        if np.random.uniform(0, 1) > 0.5:
          if np.random.uniform(0, 1) > 0.5:
            pts[:,0] *= -1.0
          else:
            pts[:,1] *= -1.0
        """

        ret_dict = {
            'filename': pair['filename'],
            'permutation_matrix': torch.from_numpy(permutation_matrix),
            'num_nodes': [pair['G1']['num_nodes'], pair['G2']['num_nodes']],
            'node_embeddings': [torch.from_numpy(x).float() for x in [pts, pair['G2']['node_embeddings']]],
            'edge_embeddings': [torch.from_numpy(x).float() for x in
                                [pair['G1']['edge_embeddings'], pair['G2']['edge_embeddings']]],
            'Kp': torch.from_numpy(pair['G1']['G_nodeaff']),
            'Ke': torch.from_numpy(pair['G1']['G_edgeaff']),
            'H': [np.asarray(x, dtype=np.float32) for x in [pair['G1']['H'], pair['G2']['H']]],
            'G': [np.asarray(x, dtype=np.float32) for x in [pair['G1']['G'], pair['G2']['G']]],
            'A1': torch.from_numpy(pair['G1']['A']),
            'A2': torch.from_numpy(pair['G2']['A']),
        }

        return ret_dict


def get_dataloader(dataset, shuffle=False):
    return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=shuffle)
