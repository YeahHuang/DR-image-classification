#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os,sys
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import pickle
import subprocess
import uuid

def aha(filename):
  if 1>0:  
    tree = ET.parse(filename)
    objs = tree.findall('object')
    num_objs = len(objs)
    if num_objs==0:
        return False

    #originally used to see whether bndbox is valid. 
    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    overlaps = np.zeros((num_objs, 3), dtype=np.float32)
    seg_areas = np.zeros((num_objs), dtype=np.float32)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
      bbox = obj.find('bndbox')
      # Make pixel indexes 0-based
      x1 = float(bbox.find('xmin').text) - 1
      y1 = float(bbox.find('ymin').text) - 1
      x2 = float(bbox.find('xmax').text) - 1
      y2 = float(bbox.find('ymax').text) - 1
      boxes[ix, :] = [x1, y1, x2, y2]
      seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)
    overlaps = scipy.sparse.csr_matrix(overlaps)
 
    #if there is more than 1 objs and bbox exists, return True
    return True

path = os.getcwd() #文件夹目录
in_path = path + "/" + "xml/"
files= os.listdir(in_path) #得到文件夹下的所有文件名称
num_lines = len(files)
idx = 0
print("overall has %d images"%num_lines)
out_path = "/Users/huangxiao/Desktop/从他们那里copy的/xml 8种颜色 12月27日版本"

idx = 0 
up_limit = 1025 #我现在只想要1025个图片
for file in files: #遍历文件夹
    if ".xml" in file: #判断是否是文件夹，不是文件夹才打开
        if aha(os.path.join(in_path,file)):
            result.append(file[0:-4]+"\n")
    idx += 1
    if idx%100==0:
        print(idx)
    if idx>=up_limit:
        break
print("overall has %d valid images"%len(result)) 
total_num = 400  
valid_filename = os.path.join(out_path, 'test.txt')
f_write = open(valid_filename, 'w') 
f_write.writelines(result[0:int(total_num/2)]) #保存入结果文件 
f_write.close()

valid_filename = os.path.join(out_path, 'train_val.txt')
f_write = open(valid_filename, 'w') 
f_write.writelines(result[int(total_num/2): int(total_num)]) #保存入结果文件 
f_write.close()   

valid_filename = os.path.join(out_path, 'train.txt')
f_write = open(valid_filename, 'w') 
f_write.writelines(result[int(total_num/8):int(total_num/8*3)]) #保存入结果文件 
f_write.close()

valid_filename = os.path.join(out_path, 'val.txt')
f_write = open(valid_filename, 'w') 
f_write.writelines(result[int(total_num/8*5):int(total_num/8*7)]) #保存入结果文件 
f_write.close()