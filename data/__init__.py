from __future__ import print_function, absolute_import

from .datasets import VC_Clothes


__datasets = {
    'vc-clothes': VC_Clothes,
}

def init_dataset(name, **kwargs):
    avai_datasets = list(__datasets.keys())
    if name not in avai_datasets:
        raise ValueError(
            'Invalid dataset name. Received "{}", '
            'but expected to be one of {}'.format(name, avai_datasets)
        )
    return __datasets[name](**kwargs)