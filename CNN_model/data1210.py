#coding:utf-8

import os
from PIL import Image
import numpy as np

trainPath = r"D:\AI\秀傳提供的bonescan檔案\已處理\train512\\"
testPath = r"D:\AI\秀傳提供的bonescan檔案\已處理\test512\\"

def changelabel(label):
    if label == "Y":
        newLabel = 1
    else:
        newLabel = 0
        
    return newLabel

def load_data():
    dim = 3 # 1: grey
    num_train_image = 800
    num_test_image = 180
    width_image = 512
    height_image = 512

    data_1 = np.empty((num_train_image,dim,width_image,height_image),dtype="float32")
    label_1 = np.empty((num_train_image,),dtype="uint8")
    data_2 = np.empty((num_test_image,dim,width_image,height_image),dtype="float32")
    label_2 = np.empty((num_test_image,),dtype="uint8")
	
    imgs_1 = os.listdir(trainPath)
    num_1 = len(imgs_1)
    for i in range(num_1):
        img_1 = Image.open(trainPath + imgs_1[i])
        arr_1 = np.asarray(img_1,dtype="float32")
        data_1[i,:,:,:] = arr_1.reshape(dim,width_image,height_image)
        label_1[i] = changelabel(imgs_1[i].split('.')[0])
    
		
    imgs_2 = os.listdir(testPath)
    num_2 = len(imgs_2)

    for i in range(num_2):
        img_2 = Image.open(testPath + imgs_2[i])
        arr_2 = np.asarray(img_2,dtype="float32")
    
        data_2[i,:,:,:] = arr_2.reshape(dim,width_image,height_image)
        label_2[i] = changelabel(imgs_2[i].split('.')[0])
	
    
    return (data_1,label_1), (data_2,label_2)

load_data()
