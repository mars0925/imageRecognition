#coding:utf-8

import os
from PIL import Image
import numpy as np

trainPath = r"D:\AI\CNN_demo\CNN_model\trainImg\\"
testPath = r"D:\AI\CNN_demo\CNN_model\testImg\\"

#讀取資料夾mnist下的42000張圖片，圖片為灰階圖，所以為1通道，圖像大小28*28
def load_data():
	data_1 = np.empty((1500,1,1024,1024),dtype="float32")
	label_1 = np.empty((1500,),dtype="uint8")
	data_2 = np.empty((250,1,1024,1024),dtype="float32")
	label_2 = np.empty((250,),dtype="uint8")
	
	imgs_1 = os.listdir(trainPath)
	num_1 = len(imgs_1)
	for i in range(num_1):
		img_1 = Image.open(trainPath + imgs_1[i])
		arr_1 = np.asarray(img_1,dtype="float32")
		data_1[i,:,:,:] = arr_1
		label_1[i] = int(imgs_1[i].split('.')[0])
		
	imgs_2 = os.listdir(testPath)
	num_2 = len(imgs_2)
	for i in range(num_2):
		img_2 = Image.open(testPath + imgs_2[i])
		arr_2 = np.asarray(img_2,dtype="float32")
		data_2[i,:,:,:] = arr_2
		label_2[i] = int(imgs_2[i].split('.')[0])
	
	print(len(data_1))
	return (data_1,label_1), (data_2,label_2)
load_data()
