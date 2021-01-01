"""General-purpose training script for image-to-image translation.
    [step 1] For Ground truth node to image
        * python data_processing.py --rule defective2img --csv ./trainAB.csv --isTrain
        
    [Step 2] For AB Features Datasets (combine Domain A&B for pix2pix inputs form train's feature full png)
        * python data_processing.py --rule AB_features --csv ./trainAB.csv --isTrain --hFlip
        
    [Step 3] For Train Datasets (spliting train's feature full png and output to root)
        * python data_processing.py --rule train --csv ./trainAB.csv --test_size 0.01 --total_size 100000
        
        
by Gary Wang
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path
from utils import Utils
import shutil
import json
import pandas as pd
from os import path
import csv
import cv2

from multiprocessing import Pool
import multiprocessing as mp
import time
from datetime import datetime

###############################################################################################  
def run_defective2img(AB_path):
    utils.run_defective2img(AB_path[0],AB_path[1],AB_path[2],AB_path[3],AB_path[4])

def run_output_AB_features_datasets(AB_list):
    utils.run_output_AB_features_datasets(AB_list[0],AB_list[1],AB_list[2])

def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)

def _build_parser():
    parser = argparse.ArgumentParser()    
    parser.add_argument('--rule', required=True, help='AB_features, train')
    parser.add_argument('--output_root', type=str, default='./datasets/outputs', help='path to images')
    parser.add_argument('--csv', default='train.csv', help='csv file (train.csv or test.csv)')
    parser.add_argument('--test_size', type=float, default=0.25, help='splite datasets to train / val')
    parser.add_argument('--isTrain', action='store_true', help='isTrain')
    parser.add_argument('--total_size', type=int, default=10000, help='train/val/test total size')
    parser.add_argument('--hFlip', action='store_true', help='hFlip')
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
    utils.init_vars(isTrain=opt.isTrain,hFlip=opt.hFlip)
    utils.init_datasets(opt.output_root)
    if opt.rule == 'AB_features':
        startTime = time.time()
        # utils.output_AB_features_datasets(csvFile=opt.csv,hFlip=opt.hFlip)  
        try:
            shutil.rmtree(utils.feature_file_path)
        except OSError as e:
            print(e)
        else:
            print("The %s directory is deleted successfully"%(utils.feature_file_path))
        os.makedirs('%s' % (utils.feature_file_path), exist_ok=True)    
        utils.delete_ipynb_checkpoints()
        train = pd.read_csv(opt.csv)
        AB_list=[]
        for index, (A_path, B_path) in enumerate(zip(train['A'], train['B'])):
            AB_list.append((index, A_path, B_path))            
        ###################
        # Muti-processing #
        ###################
        pool_size = mp.cpu_count()
        print('pool size:%d'%(10))
        pool = Pool(processes=10) # Pool() 不放參數則默認使用電腦核的數量   
        pool.map(run_output_AB_features_datasets, AB_list)
        pool.close()  
        pool.join()     
        print('Finishied %fs'%(time.time() - startTime))          
    elif opt.rule == 'train':
        utils.output_train_datasets(test_size=opt.test_size, total_size=opt.total_size)
    elif opt.rule == 'test':
        utils.output_test_datasets(csvFile=opt.csv)
    elif opt.rule == 'defective2img':    
        startTime = time.time()
        # utils.defective2img(toCsvName=opt.csv)

        trainA_path="%s/trainA"%(utils.source_data_dir_path)
        trainB_path="%s/trainB"%(utils.source_data_dir_path)
        try:
            os.remove(trainA_path)
            os.remove(trainB_path)
        except OSError as e:
            pass
        else: 
            pass        

        os.makedirs(trainA_path, exist_ok=True) 
        os.makedirs(trainB_path, exist_ok=True) 
        A = []
        B = []
        AB_path = []
        with open('%s/%s'%(utils.source_data_dir_path, utils.train_csv), encoding='UTF-8-sig') as input_file:             
            readers = csv.reader(input_file)
            for index_row, row in enumerate(readers):
                src_path = '%s/Train/%s'%(utils.source_data_dir_path, row[0])
                title_name = os.path.splitext(row[0])[0]
                A_path = '%s/trainA/%s.png'%(utils.source_data_dir_path, title_name)
                B_path = "%s/trainB/%s.png"%(utils.source_data_dir_path, title_name)
                AB_path.append((index_row,row,A_path,B_path,src_path))
                A.append(A_path)
                B.append(B_path)
                print('[%d] A:%s B:%s ... Done '%(index_row,A_path,B_path))

        # gen csv    
        csv_data = pd.DataFrame()
        csv_data['A'] = A
        csv_data['B'] = B
        csv_data.to_csv(os.path.join('./', opt.csv), index=False)
        print('%s Done'%('Train' if utils.isTrain == True else 'Test'))

        ###################
        # Muti-processing #
        ###################
        pool_size = mp.cpu_count()
        print('pool size:%d'%(10))
        pool = Pool(processes=10) # Pool() 不放參數則默認使用電腦核的數量   
        pool.map(run_defective2img, AB_path)
        pool.close()  
        pool.join()     
        print('Finishied %fs'%(time.time() - startTime))  
    