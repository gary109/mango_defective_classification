"""General-purpose training script for image-to-image translation.

    For Train Datasets
        python 02-song2img.py --output_root ../../BicycleGAN/datasets/datasets/Singing_transcription --csv ./train.csv --isTrain --shift
        
    For Test Datasets
        python 02-song2img.py --output_root ../../BicycleGAN/datasets/datasets/Singing_transcription --csv ./test.csv --shift
        
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
    parser.add_argument('--time_steps', type=float, default=1.024, help='initial learning rate for adam')
    parser.add_argument('--shift', action='store_true', help='shift time_steps in one song for features image')
    parser.add_argument('--fColorful', action='store_true', help='fColorful')
    parser.add_argument('--doCutFull', action='store_true', help='doCutFull')
    parser.add_argument('--isTrain', action='store_true', help='isTrain')
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
    utils.song2img4all(opt.csv, time_steps=opt.time_steps, shift=opt.shift, fColorful=opt.fColorful, doCutFull=opt.doCutFull, isTrain=opt.isTrain)    
    sys.exit(0)