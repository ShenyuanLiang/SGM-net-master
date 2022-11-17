import os
import sys
import cv2
import time
import torch
import numpy as np
sys.path.append('/home/skyler/Dropbox/Research/FDA/deep_learning_project/new-code/src')
from dataset import GMDataset, get_dataloader
from model import Net, get_optimizer, get_scheduler, get_criterion, hungarian, matching_accuracy
from pathlib import Path

from config import cfg

def eval_model(model, dataloader, visualize=False, dst=''):
  model.eval()

  since = time.time()
  acc_match_num = torch.zeros(1).cuda()
  acc_total_num = torch.zeros(1).cuda()
  for inputs in dataloader:
    data1, data2 = [_.cuda() for _ in inputs['node_embeddings']]
    edge1, edge2 = [_.cuda() for _ in inputs['edge_embeddings']]
    A1idx, A2idx = [_.cuda() for _ in inputs['adjacency_matrices']]
    n1_gt, n2_gt =  [_.cuda() for _ in inputs['num_nodes']]
    e1_gt, e2_gt =  [_.cuda() for _ in inputs['num_edges']] 
    perm_mat = inputs['permutation_matrix'].float().cuda()

    #data1, data2 = [_.cuda() for _ in inputs['features']]
    #edge1, edge2 = [_.cuda() for _ in inputs['edge_features']]
    #A1idx, A2idx = [_.cuda() for _ in inputs['adj_mat2list']]
    #n1_gt, n2_gt = [_.cuda() for _ in [torch.Tensor([data1.shape[1]]).int(), torch.Tensor([data2.shape[1]]).int()]]
    #perm_mat = inputs['gt_perm_mat'].cuda()

    with torch.set_grad_enabled(False):
      s_pred = model(data1, data2, edge1, edge2, A1idx, A2idx)

    s_pred_perm = hungarian(s_pred, n1_gt+e1_gt, n2_gt+e2_gt)

    _, _acc_match_num, _acc_total_num = matching_accuracy(s_pred_perm, perm_mat, n1_gt+e1_gt)
    acc_match_num += _acc_match_num
    acc_total_num += _acc_total_num

  print('ACC = {:.4f} ({}/{}) in {:.2f} seconds'.format((acc_match_num/acc_total_num).item(), int(acc_match_num.item()), int(acc_total_num.item()), time.time()-since))

  return acc_match_num / acc_total_num

def train_eval_model(model, dataloader):
  optimizer = get_optimizer(model)
  scheduler = get_scheduler(optimizer)
  criterion = get_criterion()

  start_epoch = cfg.TRAIN.START_EPOCH
  num_epochs = cfg.TRAIN.NUM_EPOCHS
  dataset_size = len(dataloader['train'].dataset)

  # create output folders
  if not Path(cfg.TRAIN.OUTPUT_PATH).exists():
    Path(cfg.TRAIN.OUTPUT_PATH).mkdir(parents=True)

  checkpoint_path = Path(cfg.TRAIN.OUTPUT_PATH) / 'models'
  if not checkpoint_path.exists():
    checkpoint_path.mkdir(parents=True)

  vis_path = Path(cfg.TRAIN.OUTPUT_PATH) / 'plots'
  if not vis_path.exists():
    vis_path.mkdir(parents=True)

  # load checkpoint if resuming training
  if cfg.TRAIN.START_EPOCH != 0:
    model_path = str(checkpoint_path / 'params_{:04}.pt'.format(start_epoch))
    model.load_state_dict(torch.load(model_path))

    optim_path = str(checkpoint_path / 'optim_{:04}.pt'.format(start_epoch))
    optimizer.load_state_dict(torch.load(optim_path))

  print('Start training...')

  best_acc = 0.0
  since = time.time()
  ll = []
  acc_all = []
  for epoch in range(start_epoch, num_epochs):
    model.train()

    epoch_loss = 0.0
    for inputs in dataloader['train']:
      data1, data2 = [_.cuda() for _ in inputs['node_embeddings']]
      edge1, edge2 = [_.cuda() for _ in inputs['edge_embeddings']]
      A1idx, A2idx = [_.cuda() for _ in inputs['adjacency_matrices']]
      n1_gt, n2_gt =  [_.cuda() for _ in inputs['num_nodes']] 
      e1_gt, e2_gt =  [_.cuda() for _ in inputs['num_edges']] 
      perm_mat = inputs['permutation_matrix'].float().cuda()

      #data1, data2 = [_.cuda() for _ in inputs['features']]
      #edge1, edge2 = [_.cuda() for _ in inputs['edge_features']]
      #A1idx, A2idx = [_.cuda() for _ in inputs['adj_mat2list']]
      #n1_gt, n2_gt = [_.cuda() for _ in [torch.Tensor([data1.shape[1]]).int(), torch.Tensor([data2.shape[1]]).int()]]
      #perm_mat = inputs['gt_perm_mat'].float().cuda()

      with torch.set_grad_enabled(True):
        optimizer.zero_grad()
        s_pred = model(data1, data2, edge1, edge2, A1idx, A2idx)
        loss = criterion(s_pred, perm_mat, n1_gt+e1_gt, n2_gt+e2_gt)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * perm_mat.size(0)

    epoch_loss = epoch_loss / dataset_size
    ll.append(epoch_loss)

    print()
    print('Epoch {:<4} Loss: {:.4f}'.format(epoch, epoch_loss))

    # generate visual results in the last epoch
    acc = eval_model(model, dataloader['test'], visualize=False, dst=vis_path)
    acc_all.append(acc.item())
    if acc > best_acc or epoch == num_epochs-1:
      best_acc = acc
      torch.save(model.state_dict(), str(checkpoint_path / 'params_{:04}.pt'.format(epoch + 1)))
      torch.save(optimizer.state_dict(), str(checkpoint_path / 'optim_{:04}.pt'.format(epoch + 1)))

    scheduler.step()
    sys.stdout.flush()

  time_elapsed = time.time() - since
  print('\nTraining complete in {:.0f}h {:.0f}m {:.0f}s'.format(time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))
  sys.stdout.flush()

  return model, ll, acc_all

def draw_pair(inputs):
  data1, data2 = inputs['node_embeddings']
  edge1, edge2 = inputs['edge_embeddings']
  A1idx, A2idx = inputs['adjacency_matrices']
  n1_gt, n2_gt = inputs['num_nodes']
  e1, e2 = inputs['num_edges']
  s_pred_perm = inputs['permutation_matrix']

  min1 = torch.min(data1[0], dim=0)[0]*100.0
  max1 = torch.max(data1[0], dim=0)[0]*100.0
  min2 = torch.min(data2[0], dim=0)[0]*100.0
  max2 = torch.max(data2[0], dim=0)[0]*100.0

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

  #cv2_imshow(img)

  bad=0
  for i in range(n1+e1):
    for j in range(n2+e2):
      if s_pred_perm[0,i,j].item() > 0.5:
        if i >= n1 and j < n2 or i < n1 and j >= n2:
          bad += 1
        p1 = v1[i]
        p2 = v2[j]
        img = cv2.line(img, p1, p2, (0,255,0), 1)
  #print('bad:', bad)

  filename = os.path.join(cfg.TRAIN.OUTPUT_PATH, 'plots', '_'.join(inputs['filename'][0].split('/')[-2:]))+'.png'
  cv2.imwrite(filename, img)

dataset = {x: GMDataset(x) for x in ('train', 'test')}
dataloader = {x: get_dataloader(dataset[x], shuffle=(x == 'train')) for x in ('train', 'test')}

print(len(dataset['train']), len(dataset['test']))

#for inputs in dataloader['train']:
#  draw_pair(inputs)

#quit()

model = Net().cuda()
model, ll, acc_all = train_eval_model(model, dataloader)

#%%
import matplotlib.pyplot as plt
plt.figure()
plt.subplot(121)
plt.plot(ll)
plt.title('training loss', fontdict = {'fontsize' : 12})
plt.subplot(122)
plt.plot(acc_all)
plt.title('testing accuracy', fontdict = {'fontsize' : 12})
plt.savefig("/home/skyler/Desktop/sim_true2.pdf", dpi = 300)
plt.show()



#%%
import sys
import scipy.linalg
sys.path.append('/home/skyler/Dropbox/Research/FDA/deep_learning_project/new-code/src')
import os
import sys
import cv2
import time
import torch
import numpy as np
import scipy.linalg
from dataset import GMDataset, get_dataloader
from model import Net, get_optimizer, get_scheduler, get_criterion, hungarian, matching_accuracy
from pathlib import Path

from config import cfg

# Implementation of "Closed-form solution of absolute orientation using orthonormal matrices" ([1] https://doi.org/10.1364/JOSAA.5.001127)
# INPUT: two arrays of corresponding points
# OUTPUT: scale factor, rotation matrix and translation vector
# PS: the obtained transformation must be applied to the second array of points
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

def draw_pair_with_gt(inputs, s_pred_perm):
  data1, data2 = inputs['node_embeddings']
  edge1, edge2 = inputs['edge_embeddings']
  A1idx, A2idx = inputs['adjacency_matrices']
  n1_gt, n2_gt = inputs['num_nodes']
  e1, e2 = inputs['num_edges']
  gt_perm = inputs['permutation_matrix']

  #print(data1[0].size(), data2[0].size())

  data_all = torch.cat((data1, data2), dim=1)
  min1 = torch.min(data_all[0], dim=0)[0]*100.0
  max1 = torch.max(data_all[0], dim=0)[0]*100.0
  min2 = torch.min(data_all[0], dim=0)[0]*100.0
  max2 = torch.max(data_all[0], dim=0)[0]*100.0

  #print(n1_gt, n2_gt)
  #print(min1, max1, min2, max2)

  width = int(max((max1[0]-min1[0]).item(), (max2[0]-min2[0]).item()))+10
  dx = 5
  height = int(max((max1[1]-min1[1]).item(), (max2[1]-min2[1]).item()))+10
  dy1 = 5
  #dy2 = int((max1[1]-min1[1]).item())+10
  dy2 = 5

  n1 = n1_gt.item()
  n2 = n2_gt.item()

  v1 = []
  v2 = []

  img = np.zeros((height,width,3), np.uint8)
  for i in range(n1):
    v1.append( (int(data1[0,i,0]*100.0-min1[0])+dx, int(data1[0,i,1]*100.0-min1[1])+dy1) )
    img = cv2.circle(img, v1[-1], 4, (255,0,0), -1)

  for i in range(n2):
    v2.append( (int(data2[0,i,0]*100.0-min2[0])+dx, int(data2[0,i,1]*100.0-min2[1])+dy2) )
    img = cv2.circle(img, v2[-1], 4, (0,255,255), -1)

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
    img = cv2.circle(img, v2[-1], 4, (0,255,255), -1)

  for i in range(n1+e1):
    for j in range(n1+e1):
      if A1idx[0,i,j] > 0.5:
        if i < n1 and j < n1:
          img = cv2.line(img, v1[i], v1[j], (255,0,0), 1)
        else:
          img = cv2.line(img, v1[i], v1[j], (255,0,0), 1)

  for i in range(n2+e2):
    for j in range(n2+e2):
      if A2idx[0,i,j] > 0.5:
        if i < n2 and j < n2:
          img = cv2.line(img, v2[i], v2[j], (0,255,255), 1)
        else:
          img = cv2.line(img, v2[i], v2[j], (0,255,255), 1)

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

  #filename = os.path.join(cfg.TRAIN.OUTPUT_PATH, 'plots', '_'.join(inputs['filename'][0].split('/')[-2:]))+'.png'
  #cv2.imwrite(filename, img)
  while cv2.waitKey(10) < 0:
    cv2.imshow('test', img)

dataset = {'test': GMDataset('test')}
dataloader = {'test': get_dataloader(dataset['test'], shuffle=False)}

model = Net().cuda()
model.load_state_dict(torch.load(os.path.join('/home/skyler/Dropbox/Research/FDA/deep_learning_project/new-code/results','models','params_0150.pt')))

for inputs in dataloader['test']:
  data1, data2 = [_.cuda() for _ in inputs['node_embeddings']]
  edge1, edge2 = [_.cuda() for _ in inputs['edge_embeddings']]
  A1idx, A2idx = [_.cuda() for _ in inputs['adjacency_matrices']]
  n1_gt, n2_gt = [_.cuda() for _ in [torch.Tensor([data1.shape[1]]).int(), torch.Tensor([data2.shape[1]]).int()]]
  e1_gt, e2_gt = [_.cuda() for _ in [torch.Tensor([edge1.shape[1]]).int(), torch.Tensor([edge2.shape[1]]).int()]]
  perm_mat = inputs['permutation_matrix'].float().cuda()

  with torch.set_grad_enabled(False):
    s_pred = model(data1, data2, edge1, edge2, A1idx, A2idx)
  s_pred_perm = hungarian(s_pred, n1_gt+e1_gt, n2_gt+e2_gt)

  pts1 = []
  pts2 = []
  for i in range(n1_gt.item()):
    for j in range(n2_gt.item()):
      if s_pred_perm[0,i,j].item() > 0.5:
        pts1.append([data1[0,i,0].item(), data1[0,i,1].item()])
        pts2.append([data2[0,j,0].item(), data2[0,j,1].item()])
  pts1 = np.asarray(pts1)
  pts2 = np.asarray(pts2)
  s, R, t = horn_absolute_orientation(pts1, pts2)

  pts2 = []
  for j in range(n2_gt.item()):
    pts2.append([data2[0,j,0].item(), data2[0,j,1].item()])
  pts2 = np.asarray(pts2)
  pts2 = s*R.dot(pts2.T).T + t
  data2 = torch.from_numpy(pts2).float().unsqueeze(0).cuda()

  with torch.set_grad_enabled(False):
    s_pred = model(data1, data2, edge1, edge2, A1idx, A2idx)
  s_pred_perm2 = hungarian(s_pred, n1_gt+e1_gt, n2_gt+e2_gt)

  pair = {
      'filename': inputs['filename'],
      'permutation_matrix': s_pred_perm,#perm_mat,
      'num_nodes': [n1_gt, n2_gt],
      'node_embeddings': [data1, data2],
      'num_edges': [e1_gt, e2_gt],
      'edge_embeddings': [edge1, edge2],
      'adjacency_matrices': [A1idx, A2idx]
  }

  #draw_pair_with_gt(pair, perm_mat)

  #perm = []
  #for i in range(n1_gt.item()):
  #  perm.append(torch.argmax(s_pred_perm[0][i]).item()+1)
  #print('/'.join(inputs['filename'][0].split('/')[-2:]), perm)
  #sys.stdout.flush()

  perm = []
  for i in range(n1_gt.item()):
    perm.append(torch.argmax(s_pred_perm[0][i]).item()+1)
  print('/'.join(inputs['filename'][0].split('/')[-2:]), perm, file=open("/home/skyler/Dropbox/Research/ElasticGraphs_Editable/Evaluation/results_true_sim.txt2", "a"))

  #break



  since = time.time()
  acc_match_num = torch.zeros(1).cuda()
  acc_total_num = torch.zeros(1).cuda()
  for inputs in dataloader['test']:
    data1, data2 = [_.cuda() for _ in inputs['node_embeddings']]
    edge1, edge2 = [_.cuda() for _ in inputs['edge_embeddings']]
    A1idx, A2idx = [_.cuda() for _ in inputs['adjacency_matrices']]
    n1_gt, n2_gt = [_.cuda() for _ in inputs['num_nodes']]
    e1_gt, e2_gt = [_.cuda() for _ in inputs['num_edges']]
    perm_mat = inputs['permutation_matrix'].float().cuda()

    # data1, data2 = [_.cuda() for _ in inputs['features']]
    # edge1, edge2 = [_.cuda() for _ in inputs['edge_features']]
    # A1idx, A2idx = [_.cuda() for _ in inputs['adj_mat2list']]
    # n1_gt, n2_gt = [_.cuda() for _ in [torch.Tensor([data1.shape[1]]).int(), torch.Tensor([data2.shape[1]]).int()]]
    # perm_mat = inputs['gt_perm_mat'].cuda()

    with torch.set_grad_enabled(False):
      s_pred = model(data1, data2, edge1, edge2, A1idx, A2idx)

    s_pred_perm = hungarian(s_pred, n1_gt + e1_gt, n2_gt + e2_gt)

    _, _acc_match_num, _acc_total_num = matching_accuracy(s_pred_perm, perm_mat, n1_gt + e1_gt)
    acc_match_num += _acc_match_num
    acc_total_num += _acc_total_num

print(
  'ACC = {:.4f} ({}/{}) in {:.2f} seconds'.format((acc_match_num / acc_total_num).item(), int(acc_match_num.item()),
                                                  int(acc_total_num.item()), time.time() - since))