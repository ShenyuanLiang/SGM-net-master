from easydict import EasyDict as edict

# Training configuration
__C = edict()
cfg = __C

# Model options
__C.MODEL = edict()
__C.MODEL.BS_ITER_NUM = 20
__C.MODEL.BS_EPSILON = 1.0e-10
__C.MODEL.NORM_ALPHA = 20.
__C.MODEL.GNN_LAYER = 4
__C.MODEL.GNN_FEAT_SIZE = 512
__C.MODEL.FC_HIDDEN_SIZE = 512

# Training options
__C.TRAIN = edict()
__C.TRAIN.START_EPOCH = 0            # Training start epoch (if not 0, will be resumed from checkpoint)
__C.TRAIN.NUM_EPOCHS = 150        # Total epochs
__C.TRAIN.LR = 1.e-4  # 2.e-5              # Start learning rate
__C.TRAIN.LR_DECAY = .5 #0.1             # Learning rate decay
__C.TRAIN.LR_STEP = [80, 120]         # Learning rate decay step (in epochs)
__C.TRAIN.MOMENTUM = 0.9      # SGD momentum

# Dataset options (PTS)
__C.TRAIN.OPTIMIZER = 'Adam'
__C.TRAIN.OUTPUT_PATH = 'new_code/results'

__C.DATASET = edict()
__C.DATASET.TYPE = 'MATLAB'
__C.DATASET.ROOT_DIR = 'new_code/graph_test'
__C.DATASET.SPLIT_INDEX = 318
__C.DATASET.SPLIT_SEED = 40 #42
__C.DATASET.NODE_EMBEDDING_SIZE = 2
__C.DATASET.EDGE_EMBEDDING_SIZE = 60

