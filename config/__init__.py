from .defaults import _C as cfg

from data.transforms import build_transforms

def model_kwargs(cfg):
    model_config = {
        'dropout': cfg.TRAIN.DROPOUT, 
        'body_feats_dim': cfg.DATA.BODY_FEATS_DIM, 
        'face_feats_dim': cfg.MODEL.FACE_FEATS_DIM
    }
    return model_config
 
def data_kwargs(cfg):
    data_config =  {
        'root': cfg.DATA.ROOT,
        'k': cfg.DATA.K,
        'num_nodes': cfg.DATA.NUM_NODES,
        'transform': None,
        'pre_transform': None
    }
    
    try:
        data_config['transform'] = build_transforms(cfg.DATA.TRANSFORM)
    except:
        print('Transform: None !')
    else:
        print('Transform:', cfg.DATA.TRANSFORM)
    
    try:
        data_config['pre_transform'] = build_transforms(cfg.DATA.PRE_TRANSFORM)
    except:
        print('pre_Transform: None !')
    else:
        print('pre_Transform:', cfg.DATA.PRE_TRANSFORM)

    return data_config