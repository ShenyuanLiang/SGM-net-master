import cv2
import math
import torch
import numpy as np
from PIL import Image
import networkx as nx
import matplotlib.pyplot as plt

from config import cfg

def show_PTS(F1, F2, perm_gt, perm_res, filename):
	size = 600
	scale = 250

	V1 = []
	V2 = []
	V2_gt = []
	color = []
	paired = set()
	for i in range(len(F1)):
		V1.append((F1[i][0],F1[i][1]))
		for j in range(len(F2)):
			if perm_res[i,j] > 0:
				V2.append((F2[j][0],F2[j][1]))
				color.append((0,255,0) if perm_gt[i,j] > 0 else (0,0,255))
				paired.add(j)
				break
		for j in range(len(F2)):
			if perm_gt[i,j] > 0:
				V2_gt.append((F2[j][0],F2[j][1]))
				break

	V2_all = []
	for i in range(len(F2)):
		V2_all.append((F2[i][0],F2[i][1]))

	V3 = []
	color3 = []
	for i in range(len(F2)):
		if i not in paired:
			V3.append((F2[i][0],F2[i][1]))
			color3.append((255,0,255) if torch.sum(perm_gt[:,i]) > 0 else (0,255,255))

	gif = []
	img = np.zeros((size,size,3), np.uint8)
	for count in range(20):
		img.fill(0)

		loop = count//10
		if loop%2 == 1:
			w = (count%10)/10.0
		else:
			w = 1.0 - (count%10)/10.0

		for i in range(len(V2_all)):
			img = cv2.circle(img, (int(V2_all[i][0]*scale+size/2.0+0.5), int(V2_all[i][1]*scale+size/2.0+0.5)), 3, (255,255,255), -1)

		for i in range(len(V3)):
			c = (int(color3[i][0]*(1.0-w) + 0.5), int(color3[i][1]*(1.0-w) + 0.5), int(color3[i][2]*(1.0-w) + 0.5))
			img = cv2.circle(img, (int(V3[i][0]*scale+size/2.0+0.5), int(V3[i][1]*scale+size/2.0+0.5)), 7, c, 2)

		for i in range(len(V1)):
			pt = ( V1[i][0]*w + V2[i][0]*(1.0-w) , V1[i][1]*w + V2[i][1]*(1.0-w) )
			c = (int(255*w + color[i][0]*(1.0-w) + 0.5), int(color[i][1]*(1.0-w) + 0.5), int(color[i][2]*(1.0-w) + 0.5))
			img = cv2.line(img, (int(pt[0]*scale+size/2.0+0.5), int(pt[1]*scale+size/2.0+0.5)), (int(V2_gt[i][0]*scale+size/2.0+0.5), int(V2_gt[i][1]*scale+size/2.0+0.5)), (255,255,255), 1)

			img = cv2.circle(img, (int(pt[0]*scale+size/2.0+0.5), int(pt[1]*scale+size/2.0+0.5)), 5, c, -1)

		gif.append(Image.fromarray(img[:,:,::-1]))
	gif[0].save(filename, save_all=True, append_images=gif[1:], loop=0, duration=100)


def show_AG(F1, A1, E1, labels1, F2, A2, E2, labels2, perm_gt, perm_res, filename):
	def drawG(G, pos, color, npoints):
		for i in range(npoints):
			G.nodes[i]['pos'] = pos[i]

		wts = [i * 0.05 for i in list(nx.get_edge_attributes(G, 'weight').values())]
		labels = nx.get_node_attributes(G, 'labels')
		pos = nx.get_node_attributes(G, 'pos')

		nx.draw(G, pos, labels=labels, width=wts, font_size=13, node_color=color)

	npoints1 = len(F1)
	npoints2 = len(F2)

	l1 = {key: value[0] for key, value in labels1.items()}
	for i in range(npoints1,npoints2):
		l1[i] = 'null'

	a1 = np.zeros([npoints2,npoints2])
	a1[:npoints1, :npoints1] = E1[A1].squeeze(-1).cpu().numpy()

	G1 = nx.from_numpy_matrix(a1)

	nx.set_node_attributes(G1, l1, 'labels')

	radius = 400
	pos1 = [(math.cos(2*math.pi/npoints2*x)*radius,math.sin(2*math.pi/npoints2*x)*radius) for x in range(0,npoints2)]

	visited = set()
	pos3 = []
	color = []
	for i in range(npoints1):
		for j in range(npoints2):
			if perm_res[i,j] > 0:
				color.append('g' if perm_gt[i,j] > 0 else 'r')
				pos3.append((pos1[j][0]+1000, pos1[j][1]))
				visited.add(j)
	for i in range(npoints2):
		if i not in visited:
			color.append('c')
			pos3.append((pos1[i][0]+1000, pos1[i][1]))

	plt.figure(figsize=(14,4))
	drawG(G1, pos1, ['c']*npoints2, npoints2)
	drawG(G1, pos3, color, npoints2)

	l2 = {key: value[0] for key, value in labels2.items()}
	a2 = E2[A2].squeeze(-1).cpu().numpy()
	G2 = nx.from_numpy_matrix(a2)
	nx.set_node_attributes(G2, l2, 'labels')
	pos2 = [(x+2000,y) for x,y in pos1]

	drawG(G2, pos2, ['c']*npoints2, npoints2)

	plt.savefig(filename)
	#plt.show()

def show_graph_pair(F1, A1, E1, labels1, F2, A2, E2, labels2, perm_gt, perm_res, filename):
	if cfg.DATASET.TYPE == 'PTS':
		show_PTS(F1, F2, perm_gt, perm_res, filename.replace('.txt','.gif'))
	elif cfg.DATASET.TYPE == 'AG':
		show_AG(F1, A1, E1, labels1, F2, A2, E2, labels2, perm_gt, perm_res, filename.replace('.txt','.png'))

