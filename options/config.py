import argparse
import os
from easydict import EasyDict as edict
import yaml


def load_config(config_path=None):
    if config_path is None:
        parser = argparse.ArgumentParser(description='Train GANs')
        parser.add_argument('config', help='train config file path')
        args = parser.parse_args()
        config_path = args.config

    with open(config_path) as fin:
        opt = yaml.load(fin, Loader=yaml.FullLoader)
        opt = edict(opt)
    for k, v in opt.items():
        print(k, ':', v)

    if 'pix2pix' in config_path or 'cycle' in config_path:
        basename = os.path.basename(config_path)
        fromto = basename.split('_')[0]
        source, dst = fromto.split('-')
        opt.name = basename[:-5]
        opt.source = source
        opt.dst = dst

    opt.name = opt.dataset + '/' + opt.name
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
    return opt

def load_config_single(config_path=None):
    if config_path is None:
        parser = argparse.ArgumentParser(description='Train a detector')
        parser.add_argument('config', help='train config file path')
        args = parser.parse_args()
        config_path = args.config
    with open(config_path) as fin:
        opt = yaml.load(fin, Loader=yaml.FullLoader)
        opt = edict(opt)
    for k, v in opt.items():
        print(k, ':', v)
    
    basename = os.path.basename(config_path)
    fromto = basename.split('_')[0]
    source, dst = fromto.split('-')
    opt.name = basename[:-5]
    opt.source = source
    opt.dst = dst
    
    opt.name = opt.dataset + '/' + opt.name
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
    return opt