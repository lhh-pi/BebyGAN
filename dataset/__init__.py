from easydict import EasyDict as edict
import importlib
import os

from utils.common import scandir


dataset_root = os.path.dirname(os.path.abspath(__file__))
dataset_filenames = [
        os.path.splitext(os.path.basename(v))[0] for v in scandir(dataset_root)
        if v.endswith('_dataset.py')
]
_dataset_modules = [
        importlib.import_module(f'dataset.{file_name}')
        for file_name in dataset_filenames
]


class DATASET:
    LEGAL = ['DIV2K', 'Flickr2K', 'Set5', 'Set14', 'BSDS100', 'Urban100', 'Manga109']

    # training dataset
    DIV2K = edict()
    DIV2K.TRAIN = edict()
    DIV2K.TRAIN.HRx2 = '../../datasets/DIV2K/DIV2K_train_HR_sub'  # 32208
    DIV2K.TRAIN.HRx3 = '../../datasets/DIV2K/DIV2K_train_HR_sub'  # 32208
    DIV2K.TRAIN.HRx4 = '../../datasets/DIV2K/DIV2K_train_HR_sub'  # 32208
    DIV2K.TRAIN.LRx2 = '../../datasets/DIV2K/DIV2K_train_LR_bicubic/X2_sub'
    DIV2K.TRAIN.LRx3 = '../../datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'
    DIV2K.TRAIN.LRx4 = '../../datasets/DIV2K/DIV2K_train_LR_bicubic/X4_sub'

    Flickr2K = edict()
    Flickr2K.TRAIN = edict()
    Flickr2K.TRAIN.HRx2 = None
    Flickr2K.TRAIN.HRx3 = None
    Flickr2K.TRAIN.HRx4 = None
    Flickr2K.TRAIN.LRx2 = None
    Flickr2K.TRAIN.LRx3 = None
    Flickr2K.TRAIN.LRx4 = None

    # testing dataset
    Set5 = edict()
    Set5.VAL = edict()
    Set5.VAL.HRx2 = '../../datasets/Set5/GTmod12'
    Set5.VAL.HRx3 = '../../datasets/Set5/GTmod12'
    Set5.VAL.HRx4 = '../../datasets/Set5/GTmod12'
    Set5.VAL.LRx2 = '../../datasets/Set5/LRbicx2'
    Set5.VAL.LRx3 = '../../datasets/Set5/LRbicx3'
    Set5.VAL.LRx4 = '../../datasets/Set5/LRbicx4'

    Set14 = edict()
    Set14.VAL = edict()
    Set14.VAL.HRx2 = None
    Set14.VAL.HRx3 = None
    Set14.VAL.HRx4 = None
    Set14.VAL.LRx2 = None
    Set14.VAL.LRx3 = None
    Set14.VAL.LRx4 = None

    BSDS100 = edict()
    BSDS100.VAL = edict()
    BSDS100.VAL.HRx2 = '/data/liwenbo/datasets/benchmark_SR/BSDS100/HR/modX2'
    BSDS100.VAL.HRx3 = '/data/liwenbo/datasets/benchmark_SR/BSDS100/HR/modX3'
    BSDS100.VAL.HRx4 = '/data/liwenbo/datasets/benchmark_SR/BSDS100/HR/modX4'
    BSDS100.VAL.LRx2 = '/data/liwenbo/datasets/benchmark_SR/BSDS100/LR_bicubic/X2'
    BSDS100.VAL.LRx3 = '/data/liwenbo/datasets/benchmark_SR/BSDS100/LR_bicubic/X3'
    BSDS100.VAL.LRx4 = '/data/liwenbo/datasets/benchmark_SR/BSDS100/LR_bicubic/X4'

    Urban100 = edict()
    Urban100.VAL = edict()
    Urban100.VAL.HRx2 = None
    Urban100.VAL.HRx3 = None
    Urban100.VAL.HRx4 = None
    Urban100.VAL.LRx2 = None
    Urban100.VAL.LRx3 = None
    Urban100.VAL.LRx4 = None

    Manga109 = edict()
    Manga109.VAL = dict()
    Manga109.VAL.HRx2 = None
    Manga109.VAL.HRx3 = None
    Manga109.VAL.HRx4 = None
    Manga109.VAL.LRx2 = None
    Manga109.VAL.LRx3 = None
    Manga109.VAL.LRx4 = None


def get_dataset(config):
    dataset_type = config.TYPE
    dataset_cls = None
    for module in _dataset_modules:
        dataset_cls = getattr(module, dataset_type, None)
        if dataset_cls is not None:
            break
    if dataset_cls is None:
        raise ValueError(f'Dataset {dataset_type} is not found.')

    hr_paths = []
    lr_paths = []
    D = DATASET()

    for dataset, split in zip(config.DATASETS, config.SPLITS):
        if dataset not in D.LEGAL or split not in eval('D.%s' % dataset):
            raise ValueError('Illegal dataset.')
        hr_paths.append(eval('D.%s.%s.HRx%d' % (dataset, split, config.SCALE)))
        lr_paths.append(eval('D.%s.%s.LRx%d' % (dataset, split, config.SCALE)))

    return dataset_cls(hr_paths, lr_paths, config)

