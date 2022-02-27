from __future__ import absolute_import

from .sef import SEF
from .ablation import Linear, GCNBody, GCNFace, Linear_GCNBody, Linear_GCNFace, GCNBody_GCNFace

__model_factory = {
    'sef': SEF,
    'linear': Linear,
    'gcnbody': GCNBody,
    'gcnface': GCNFace,
    'linear-gcnbody': Linear_GCNBody,
    'linear-gcnface': Linear_GCNFace,
    'gcnbody-gcnface': GCNBody_GCNFace,
}

def show_avai_models():
    "Displays available models."
    print(list(__model_factory.keys()))

def build_model(name, **kwargs):
    avai_models = list(__model_factory.keys())
    if name not in avai_models:
        raise KeyError(
            'Unknown model: {}. Must be one of {}'.format(name, avai_models)
        )
    return __model_factory[name](**kwargs)