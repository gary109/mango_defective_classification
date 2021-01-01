"""General-purpose training script for image-to-image translation.
    For Predict Test Datasets
        python 04-predict_datasets.py --csv ./test.csv --shift
        
by Gary Wang
"""
import sys
import subprocess
import argparse
from pathlib import Path
from utils import Utils

def _build_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--output_root', required=True, help='path to images(../../BicycleGAN/datasets/datasets/Singing_transcription)')
    parser.add_argument('--csv', required=True, default='train.csv', help='csv file (train.csv or test.csv)')
    parser.add_argument('--time_steps', type=float, default=1.024, help='time_steps for shift scale')
    parser.add_argument('--pixel_interval', type=float, default=0.016, help='time/pixel')
    parser.add_argument('--time_interval', type=float, default=4.096, help='image size 256x256 total esaplsed time')
    parser.add_argument('--test_size', type=float, default=0.05, help='splite datasets to train / val')
    parser.add_argument('--shift', action='store_true', help='shift time_steps in one song for features image')
    return parser

def print_options(parser,opt):
    """Print and save options
    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)
            
if __name__ == '__main__':
    parser = _build_parser()
    opt, _ = parser.parse_known_args()
    
    print_options(parser,opt)
    utils = Utils()
    utils.init_datasets(opt.output_root)
    if opt.rule == 'AB_features':
        utils.output_AB_features_datasets(csvFile=opt.csv, time_steps=opt.time_steps, shift=opt.shift, time_interval=opt.time_interval, pixel_interval=opt.pixel_interval)  
    elif opt.rule == 'train':
        utils.output_train_datasets(test_size=opt.test_size)
    elif opt.rule == 'test':
        utils.output_test_datasets(csvFile=opt.csv)
    sys.exit(0)