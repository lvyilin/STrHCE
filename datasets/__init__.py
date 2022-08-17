from torchvision.datasets import ImageNet, CIFAR10, CIFAR100, SVHN

from datasets import aircraft, aircraft_hier, dogs_hier, aircraft_hier2, nabirds_hier, cub2011hier2
from datasets import cub2011
from datasets import cub2011hier
from datasets import dogs
from datasets import nabirds

_dataset_class_map = {
    'aircraft': aircraft.Aircraft,
    'cub2011': cub2011.Cub2011,
    'dogs': dogs.Dogs,
    'nabirds': nabirds.NABirds,
    'imagenet': ImageNet,
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
    'svhn': SVHN,
    'cub2011hier': cub2011hier.Cub2011,
    'cub2011hier2': cub2011hier2.Cub2011,
    'aircraft_hier': aircraft_hier.Aircraft,
    'aircraft_hier2': aircraft_hier2.Aircraft,
    'dogs_hier': dogs_hier.Dogs,
    'nabirds_hier': nabirds_hier.NABirds,

}
_dataset_class_number_map = {
    'aircraft': 100,
    'cub2011': 200,
    'dogs': 120,
    'nabirds': 555,
    'cub2011hier': 200,
    'cub2011hier2': 200,
    'aircraft_hier': 100,
    'aircraft_hier2': 100,
    'nabirds_hier': 555,
}

_dataset_parent_class_number_map = {
    'cub2011hier': 122,
    'cub2011hier2': 122,
    'aircraft_hier': 70,
    'aircraft_hier2': 70,
    'dogs_hier': 72,
    'nabirds_hier': 404,

}
_dataset_pparent_class_number_map = {
    'aircraft_hier2': 30,
    'cub2011hier2': 37,
}


def get_dataset(dataset_name, **kwargs):
    if dataset_name.endswith('imagenet'):
        split = 'train' if kwargs['train'] else 'val'
        del kwargs['train']
        kwargs['split'] = split
    elif dataset_name == 'svhn':
        split = 'train' if kwargs['train'] else 'test'
        del kwargs['train']
        kwargs['split'] = split
    elif dataset_name == 'cub2011a':
        kwargs['specific_classes'] = kwargs['in_dist']
    dataset = _dataset_class_map[dataset_name](**kwargs)
    return dataset


def get_dataset_class_number(dataset_name):
    return _dataset_class_number_map[dataset_name]


def get_dataset_parent_class_number(dataset_name):
    return _dataset_parent_class_number_map[dataset_name]


def get_dataset_pparent_class_number(dataset_name):
    return _dataset_pparent_class_number_map[dataset_name]
