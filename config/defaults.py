from yacs.config import CfgNode as CN

_C = CN()

_C.MODEL = CN()
_C.MODEL.NAME = "sef"
_C.MODEL.WEIGHT = ""

_C.DATA = CN()
_C.DATA.DATASET = 'vc-clothes'
_C.DATA.ROOT = '/data/vc-clothes'
_C.DATA.K = 100
_C.DATA.NUM_NODES = 300
_C.DATA.BODY_FEATS_DIM = 2048
_C.DATA.FACE_FEATS_DIM = 512
_C.DATA.TRANSFORM = []
_C.DATA.PRE_TRANSFORM = ['NormalizeNodes']


_C.TRAIN = CN()
_C.TRAIN.LR = 1e-3
_C.TRAIN.WEIGHT_DECAY = 0.0
_C.TRAIN.DROPOUT = 0.2
_C.TRAIN.MAX_EPOCH = 20
_C.TRAIN.BATCH_SIZE = 24
_C.TRAIN.PRINT_FREQ = 20
_C.TRAIN.SEED = 123

_C.TEST = CN()
_C.TEST.EVAL_FREQ = 1

_C.OUTPUT_DIR = "./logs/"
_C.LOG_ROOT = "./logs/"
_C.CHECKPOINTS_ROOT = "/data4/xieqk/megcn/logs/"
_C.DIR_NAME = "demo"
