import os
import sys
import cv2
import time
import torch
import numpy as np
from dataset import GMDataset, get_dataloader
from model import Net, get_optimizer, get_scheduler, get_criterion, hungarian, matching_accuracy
from pathlib import Path

from config import cfg

def draw_pair_with_gt(inputs, s_pred_perm):
  data1, data2 = inputs['node_embeddings']
  edge1, edge2 = inputs['edge_embeddings']
  A1idx, A2idx = inputs['adjacency_matrices']
  n1_gt, n2_gt = inputs['num_nodes']
  e1, e2 = inputs['num_edges']
  gt_perm = inputs['permutation_matrix']

  #print(data1[0].size(), data2[0].size())

  min1 = torch.min(data1[0], dim=0)[0]*100.0
  max1 = torch.max(data1[0], dim=0)[0]*100.0
  min2 = torch.min(data2[0], dim=0)[0]*100.0
  max2 = torch.max(data2[0], dim=0)[0]*100.0

  #print(n1_gt, n2_gt)
  #print(min1, max1, min2, max2)

  width = int(max((max1[0]-min1[0]).item(), (max2[0]-min2[0]).item()))+10
  dx = 5
  height = int((max1[1]-min1[1]).item())+int((max2[1]-min2[1]).item())+15
  dy1 = 5
  dy2 = int((max1[1]-min1[1]).item())+10

  n1 = n1_gt.item()
  n2 = n2_gt.item()

  v1 = []
  v2 = []

  img = np.zeros((height,width,3), np.uint8)
  for i in range(n1):
    v1.append( (int(data1[0,i,0]*100.0-min1[0])+dx, int(data1[0,i,1]*100.0-min1[1])+dy1) )
    img = cv2.circle(img, v1[-1], 4, (255,255,255), -1)

  for i in range(n2):
    v2.append( (int(data2[0,i,0]*100.0-min2[0])+dx, int(data2[0,i,1]*100.0-min2[1])+dy2) )
    img = cv2.circle(img, v2[-1], 4, (255,255,255), -1)

  for i in range(n1, n1+e1):
    p = q = -1
    for j in range(n1):
      if A1idx[0,i,j] > 0.5:
        p = j
      if A1idx[0,j,i] > 0.5:
        q = j
    x = (v1[p][0]+v1[q][0])//2+3
    y = (v1[p][1]+v1[q][1])//2+3
    v1.append((x, y))
    img = cv2.circle(img, v1[-1], 4, (255,0,0), -1)

  for i in range(n2, n2+e2):
    p = q = -1
    for j in range(n2):
      if A2idx[0,i,j] > 0.5:
        p = j
      if A2idx[0,j,i] > 0.5:
        q = j
    x = (v2[p][0]+v2[q][0])//2+3
    y = (v2[p][1]+v2[q][1])//2+3
    v2.append((x, y))
    img = cv2.circle(img, v2[-1], 4, (255,0,0), -1)

  for i in range(n1+e1):
    for j in range(n1+e1):
      if A1idx[0,i,j] > 0.5:
        if i < n1 and j < n1:
          img = cv2.line(img, v1[i], v1[j], (0,255,255), 1)
        else:
          img = cv2.line(img, v1[i], v1[j], (255,0,255), 1)

  for i in range(n2+e2):
    for j in range(n2+e2):
      if A2idx[0,i,j] > 0.5:
        if i < n2 and j < n2:
          img = cv2.line(img, v2[i], v2[j], (0,255,255), 1)
        else:
          img = cv2.line(img, v2[i], v2[j], (255,0,255), 1)

  bad=0
  for i in range(n1+e1):
    for j in range(n2+e2):
      if s_pred_perm[0,i,j].item() > 0.5:
        if i >= n1 and j < n2 or i < n1 and j >= n2:
          bad += 1
        p1 = v1[i]
        p2 = v2[j]
        if gt_perm[0,i,j].item() > 0.5:
          img = cv2.line(img, p1, p2, (0,255,0), 1)
        else:
          img = cv2.line(img, p1, p2, (0,0,255), 1)
  #print('bad:', bad)

  print()
  filename = os.path.join(cfg.TRAIN.OUTPUT_PATH, 'plots', '_'.join(inputs['filename'][0].split('/')[-2:]))+'.png'
  cv2.imwrite(filename, img)

dataset = {x: GMDataset(x) for x in ('train', 'test')}
dataloader = {x: get_dataloader(dataset[x], shuffle=(x == 'train')) for x in ('train', 'test')}

print(len(dataset['train']), len(dataset['test']))

#for inputs in dataloader['train']:
#  draw_pair(inputs)

#quit()

model = Net().cuda()
model.load_state_dict(torch.load(os.path.join(cfg.TRAIN.OUTPUT_PATH,'models','params_0049.pt')))

for inputs in dataloader['test']:
  data1, data2 = [_.cuda() for _ in inputs['node_embeddings']]
  edge1, edge2 = [_.cuda() for _ in inputs['edge_embeddings']]
  A1idx, A2idx = [_.cuda() for _ in inputs['adjacency_matrices']]
  n1_gt, n2_gt = [_.cuda() for _ in [torch.Tensor([data1.shape[1]]).int(), torch.Tensor([data2.shape[1]]).int()]]
  e1_gt, e2_gt = [_.cuda() for _ in [torch.Tensor([edge1.shape[1]]).int(), torch.Tensor([edge2.shape[1]]).int()]]
  perm_mat = inputs['permutation_matrix'].float().cuda()

  since = time.time()
  with torch.set_grad_enabled(False):
    s_pred = model(data1, data2, edge1, edge2, A1idx, A2idx)
  s_pred_perm = hungarian(s_pred, n1_gt+e1_gt, n2_gt+e2_gt)

  pair = {
      'filename': inputs['filename'],
      'permutation_matrix': s_pred_perm,#perm_mat,
      'num_nodes': [n1_gt, n2_gt],
      'node_embeddings': [data1, data2],
      'num_edges': [e1_gt, e2_gt],
      'edge_embeddings': [edge1, edge2],
      'adjacency_matrices': [A1idx, A2idx]
  }

  draw_pair_with_gt(pair, perm_mat)

  perm = []
  for i in range(n1_gt.item()):
    perm.append(torch.argmax(s_pred_perm[0][i]).item()+1)
  print('/'.join(inputs['filename'][0].split('/')[-2:]), perm)
  sys.stdout.flush()

