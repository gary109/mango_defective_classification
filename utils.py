import librosa, librosa.display
from bitstring import BitArray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import IPython.display as ipd
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.transform import rescale, resize, downscale_local_mean
import cv2
import csv
from glob import glob
from time import time
import shutil
import os, time
import scipy
from scipy import signal
from scipy import stats
from scipy.misc import imsave
import datetime
import math
from pathlib import Path
import youtube_dl
import argparse
import sys
import mido
import json
import os.path
from os import path
import seaborn as sns

from madmom.audio.filters import LogarithmicFilterbank
from madmom.features.onsets import SpectralOnsetProcessor
from madmom.audio.signal import normalize

import parselmouth

from midiutil.MidiFile import MIDIFile
from pandas.core.frame import DataFrame

from multiprocessing import Pool
import multiprocessing as mp

sns.set() # Use seaborn's default style to make attractive graphs
       
class Utils:        
    def __init__(self):
        pass
        
    def init_vars(self, isTrain=True, hFlip=True):   
        self.hFlip = hFlip
        self.isTrain = isTrain
        print("isTrain=%d"%(self.isTrain))    

    def run_output_AB_features_datasets(self,index,A_path,B_path):
        inputs_A = cv2.imread(A_path)
        inputs_B = cv2.imread(B_path)
        A_basename = os.path.splitext(os.path.basename(A_path))[0]
        B_basename = os.path.splitext(os.path.basename(B_path))[0]
        save_path = '/'.join((self.feature_file_path, '{}_{}.png'.format(A_basename, B_basename)))
        data = np.hstack((inputs_A,inputs_B)) # Domain A & B
        rs_data = cv2.resize(data, (256*2, 256), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(save_path, rs_data)   
        print('[%d] Combine A(%s) + B(%s) ... %s Done'%(index, A_basename, B_basename, save_path))
        if self.hFlip:
            h_flip_A = cv2.flip(inputs_A, 1)
            h_flip_B = cv2.flip(inputs_B, 1)
            save_path = '/'.join((self.feature_file_path, 'h{}_h{}.png'.format(A_basename, B_basename)))
            data = np.hstack((h_flip_A,h_flip_B)) # Domain A & B
            rs_data = cv2.resize(data, (256*2, 256), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(save_path, data)
            print('[%d] Combine hA(%s) + hB(%s) ... %s Done'%(index, A_basename, B_basename, save_path))
                
    def output_AB_features_datasets(self, csvFile):
        ###################################################################################################
        try:
            shutil.rmtree(self.feature_file_path)
        except OSError as e:
            print(e)
        else:
            print("The %s directory is deleted successfully"%(self.feature_file_path))
        os.makedirs('%s' % (self.feature_file_path), exist_ok=True)    
        self.delete_ipynb_checkpoints()
        ###################################################################################################
        train = pd.read_csv(csvFile)
        ###################################################################################################
        for index, (A_path, B_path) in enumerate(zip(train['A'], train['B'])):
            self.run_output_AB_features_datasets(index,A_path, B_path)
        print("=== Done ===")
        
    def output_test_datasets(self, csvFile, who=None):
        test = pd.read_csv(csvFile)
        imgs_MP3 = []
        count = 1
        for _path, _songs in zip(test['paths'],test['songs']):
            mp3_path = os.path.join(_path, _songs)
            dirname = os.path.dirname(mp3_path)
            features_path = '%s/features'%(os.path.dirname(mp3_path))
            basename = os.path.basename(mp3_path)
            imgname = '%s_feature_full.png'%(os.path.splitext(basename)[0])
            mp3_img_path = os.path.join(_path, imgname)
            if path.exists(mp3_img_path) == True and self.renew_all == True:
                print('[%d] %s'%(count, mp3_img_path))
                imgs_MP3.append(mp3_img_path)
                count += 1
            elif path.exists(mp3_img_path) == True and self.renew_all == False and path.exists(features_path) == False:
                print('[%d] %s'%(count, mp3_img_path))
                imgs_MP3.append(mp3_img_path)
                count += 1
            else:
                print(" Not found MP3 %s"%(mp3_img_path))   
        if who == None:
            self.cut_features_image(imgs_MP3)
        else:
            self.cut_features_image([imgs_MP3[who]])
        print("=== Done ===")
        
    def output_train_datasets(self, test_size=0.1, total_size=10000):
        import random
        startTime = time.time()
        
        image_path_list = []
        for file_path in glob('{}/*.png'.format(self.feature_file_path)):
            image_path_list.append(file_path)
            
        if len(image_path_list) < total_size:
            total_size = len(image_path_list)
            
        data_list = shuffle(image_path_list)  
        
        _train_size = int(total_size*(1-test_size))
        _test_size = int((total_size-_train_size)/2)
        _val_size = total_size-(_train_size+_test_size)
        
        train_list = data_list[0:_train_size]
        test_list = data_list[_train_size:_train_size+_test_size]
        val_list = data_list[_train_size+_test_size:]
        
        print("train:%d test:%d val:%d"%(len(train_list), len(test_list), len(val_list)))

        path_list = [self.train_file_path, self.val_file_path, self.test_file_path]
        for pa in path_list:
            try:
                shutil.rmtree(pa)
            except OSError as e:
                print(e)
            else:
                print("The %s directory is deleted successfully"%(pa))
            os.makedirs('%s' % (pa), exist_ok=True)
            
        self.delete_ipynb_checkpoints()
        
        
        # Muti-processing
        print('### Creating Train/Val/Test Datasets ###')  
        pool_size = 5#mp.cpu_count()
        print('pool size:%d'%(pool_size))
        pool = Pool(processes=pool_size) # Pool() 不放參數則默認使用電腦核的數量   
        # pool.map(self.run_copy_train_file, train_list)
        # pool.map(self.run_copy_val_file, val_list)
        # pool.map(self.run_copy_test_file, test_list)
        pool.map(self.run_move_train_file, train_list)
        pool.map(self.run_move_val_file, val_list)
        pool.map(self.run_move_test_file, test_list)
        pool.close()  
        pool.join()     
        print('Finishied %fs'%(time.time() - startTime))  
                
    def run_copy_train_file(self, __path):
        basename = os.path.basename(__path)
        save_path = '/'.join((self.train_file_path, basename))
        print('[COPY Train Datasets] %s -> %s '%(__path, save_path))
        shutil.copy(__path, save_path)
        
    def run_copy_val_file(self, __path):
        basename = os.path.basename(__path)
        save_path = '/'.join((self.val_file_path, basename))
        print('[COPY Val Datasets] %s -> %s '%(__path, save_path))
        shutil.copy(__path, save_path)
        
    def run_copy_test_file(self, __path):
        basename = os.path.basename(__path)
        save_path = '/'.join((self.test_file_path, basename))
        print('[COPY Test Datasets] %s -> %s '%(__path, save_path))
        shutil.copy(__path, save_path)

    def run_move_train_file(self, __path):
        basename = os.path.basename(__path)
        save_path = '/'.join((self.train_file_path, basename))
        print('[MOVE Train Datasets] %s -> %s '%(__path, save_path))
        shutil.move(__path, save_path)
        
    def run_move_val_file(self, __path):
        basename = os.path.basename(__path)
        save_path = '/'.join((self.val_file_path, basename))
        print('[MOVE Val Datasets] %s -> %s '%(__path, save_path))
        shutil.move(__path, save_path)
        
    def run_move_test_file(self, __path):
        basename = os.path.basename(__path)
        save_path = '/'.join((self.test_file_path, basename))
        print('[MOVE Test Datasets] %s -> %s '%(__path, save_path))
        shutil.move(__path, save_path)
        
    def init_datasets(self,output_root):
        self.defectiveInfo = {
            '不良-著色不佳':(128,128,128),
            '不良-機械傷害':(64,64,64),
            '不良-乳汁吸附':(32,32,32), 
            '不良-炭疽病'  :(16,16,16),                       
            '不良-黑斑病'  :(8,8,8),
        }
        
        self.original_dir = 'original'
        self.label_csv = 'label.csv'
        self.train_csv = 'train.csv'
        self.Disease_sample = 'Disease_sample'
        self.C2_TrainDev = 'C2_TrainDev'
        self.AIcup_testset_ok = 'AIcup_testset_ok'
        self.train_dir = 'train'
        self.val_dir = 'val'
        self.test_dir = 'test'
        self.features_dir = 'features'
#         self.source_data_dir_path = './datasets/%s/%s'%(self.original_dir,self.Disease_sample) 
        self.source_data_dir_path = './datasets/%s/%s'%(self.original_dir,self.C2_TrainDev) 
        self.source_testdata_dir_path = './datasets/%s/%s'%(self.original_dir,self.AIcup_testset_ok)       
        self.feature_file_path = '/'.join((output_root, self.features_dir)) 
        self.train_file_path = '/'.join((output_root, self.train_dir))
        self.val_file_path = '/'.join((output_root, self.val_dir))
        self.test_file_path = '/'.join((output_root, self.test_dir))
        self.delete_ipynb_checkpoints()

    def run_defective2img(self, index_row,row,A_path,B_path,src_path):
        inputs = cv2.imread(src_path)
        cv2.imwrite(A_path, inputs)
        height, width, channels = inputs.shape                    
        img_B1 = np.zeros((height,width,3), np.uint8)       
        cv2.imwrite(B_path, img_B1)
        for index,data in enumerate(row[1::5]):
            if data != '':
                ###########################
                x = int(float(row[index*5+1]))
                y = int(float(row[index*5+2]))
                w = int(float(row[index*5+3]))
                h = int(float(row[index*5+4]))
                defective_name = row[index*5+5]                
                ###########################  
                inputs = cv2.imread(B_path)
                height, width, channels = inputs.shape
                img_B2 = np.zeros((height,width,3), np.uint8)
                cv2.rectangle(img_B2, (x, y), (x+w,y+h), self.defectiveInfo[defective_name], -1)
                img_B = inputs | img_B2
                cv2.imwrite(B_path, img_B)
        print('[%d] A:%s B:%s ... Done '%(index_row,A_path,B_path))

    def defective2img(self, toCsvName): 
        trainA_path="%s/trainA"%(self.source_data_dir_path)
        trainB_path="%s/trainB"%(self.source_data_dir_path)
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
        with open('%s/%s'%(self.source_data_dir_path, self.train_csv), encoding='UTF-8-sig') as input_file:             
            readers = csv.reader(input_file)
            for index_row, row in enumerate(readers):
                src_path = '%s/Train/%s'%(self.source_data_dir_path, row[0])
                title_name = os.path.splitext(row[0])[0]
                A_path = '%s/trainA/%s.png'%(self.source_data_dir_path, title_name)
                B_path = "%s/trainB/%s.png"%(self.source_data_dir_path, title_name)                
                self.run_defective2img(index_row,row,A_path,B_path,src_path)
                A.append(A_path)
                B.append(B_path)                

        # gen csv    
        csv_data = pd.DataFrame()
        csv_data['A'] = A
        csv_data['B'] = B
        csv_data.to_csv(os.path.join('./', toCsvName), index=False)
        print('%s Done'%('Train' if self.isTrain == True else 'Test'))
                               
    def delete_ipynb_checkpoints(self):
        # delete all .ipynb_checkpoints dir
        for filename in Path(os.getcwd()).glob('**/*.ipynb_checkpoints'):
            try:
                shutil.rmtree(filename)
            except OSError as e:
                print(e)
            else: 
                print("The %s is deleted successfully" % (filename))

    