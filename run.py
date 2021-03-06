import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn

import data
from experiment import Experiment

import faulthandler
faulthandler.enable()

"""
nohup python run.py -- lr 1e-3 -- num_workers 4 -- batch_size 4 -- epochs 60 -- cuda -- ngpu 1 -- refs 2 -- patch_size 35 -- patch_stride 30 -- test_patch 75 -- pretrained encoder.pth -- save_dir out -- train_dir data/train -- val_dir data/val -- test_dir data/val & > out.log &
"""

# 获取模型运行时必须的一些参数
parser = argparse.ArgumentParser(description = 'Acquire some parameters for fusion restore')
parser.add_argument(' -- lr', type = float, default = 1e-3, 
                    help = 'the initial learning rate')
parser.add_argument(' -- batch_size', type = int, default = 32, 
                    help = 'input batch size for training')
parser.add_argument(' -- epochs', type = int, default = 30, 
                    help = 'number of epochs to train')
parser.add_argument(' -- cuda', action = 'store_true', help = 'enables cuda')
parser.add_argument(' -- ngpu', type = int, default = 1, help = 'number of GPUs to use')
parser.add_argument(' -- num_workers', type = int, default = 0, help = 'number of threads to load data')
parser.add_argument(' -- save_dir', type = Path, default = Path('.'), 
                    help = 'the output directory')
parser.add_argument(' -- pretrained', type = Path, help = 'the path of the pretained encoder')

# 获取对输入数据进行预处理时的一些参数
parser.add_argument(' -- refs', type = int, help = 'the reference data counts for fusion')
parser.add_argument(' -- train_dir', type = Path, default = (data.data_dir / 'train'), 
                    help = 'the training data directory')
parser.add_argument(' -- val_dir', type = Path, default = (data.data_dir / 'val'), 
                    help = 'the validation data directory')
parser.add_argument(' -- test_dir', type = Path, default = (data.data_dir / 'val'), 
                    help = 'the test data directory')
parser.add_argument(' -- image_size', type = int, nargs = '+', default = [300, 300], 
                    help = 'the size of the coarse image (width, height)')
parser.add_argument(' -- patch_size', type = int, nargs = '+', default = 10, 
                    help = 'the coarse image patch size for training restore')
parser.add_argument(' -- patch_stride', type = int, nargs = '+', default = 5, 
                    help = 'the coarse patch stride for image division')
parser.add_argument(' -- test_patch', type = int, nargs = '+', default = 50, 
                    help = 'the coarse image patch size for fusion test')
opt = parser.parse_args()

torch.manual_seed(2019)
if not torch.cuda.is_available():
    opt.cuda = False
if opt.cuda:
    torch.cuda.manual_seed_all(2019)
    cudnn.benchmark = True
    cudnn.deterministic = True

if __name__ == '__main__':
    experiment = Experiment(opt)
    if opt.epochs > 0:
        experiment.train(opt.train_dir, opt.val_dir, 
                         opt.patch_size, opt.patch_stride, opt.batch_size, 
                         opt.refs, num_workers = opt.num_workers, epochs = opt.epochs)
    experiment.test(opt.test_dir, opt.test_patch, opt.refs, 
                    num_workers = opt.num_workers)

