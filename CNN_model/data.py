#coding:utf-8

import os
from PIL import Image
import numpy as np

#彩色圖片輸入,將channel 1 改成 3，data[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]

trainPath = r"E:\Share\AutoML_8成準確率資料\train\\"
testPath = r"E:\Share\AutoML_8成準確率資料\test\\"

# 資料夾內照片種類以及張數
def showPicNumber(dirName, label):

	labelset = set(label)
	labels = list(label)

	for x in labelset:
	    pass
	    print(dirName ," 標籤 ", x," ", labels.count(x))
	    
def load_data():
	imgs_train = os.listdir(trainPath)#列出訓練資料檔案夾內所有檔案名稱
	imgs_test = os.listdir(testPath)#列出測試資料檔案夾內所有檔案名稱
	firstImage = Image.open(trainPath + imgs_train[0])#開啟第一張圖片
	pixel = firstImage.size[0]#照片尺寸大小
	num_train = len(imgs_train)#訓練檔案夾有多少圖片
	num_test = len(imgs_test)#訓練檔案夾有多少圖片


	data_1 = np.empty((num_train,pixel,pixel,1),dtype="uint8")
	label_1 = np.empty((num_train,),dtype="uint8")
	data_2 = np.empty((num_test,1,pixel,pixel),dtype="uint8")
	label_2 = np.empty((num_test,),dtype="uint8")
	
	for i in range(num_train):
		img_train = Image.open(trainPath + imgs_train[i])
		arr_1 = np.array(img_train)
		x_train4D = x_train.reshape(x_train.shape[0],28,28,1).astype("float32")


		data_1[i,:,:,:] = [arr_1[:,:,0],arr_1[:,:,1],arr_1[:,:,2]]
		label_1[i] = int(imgs_train[i].split('.')[0])
		# arr_1 = np.asarray(img_train,dtype="float32")
		# data_1[i,:,:,:] = arr_1
		# label_1[i] = int(imgs_train[i].split('.')[0])

		# 調整label

		if label_1[i] == 3:
			label_1[i] = 0
		else:
			label_1[i] = 1
		
	for i in range(num_test):
		img_test = Image.open(testPath + imgs_test[i])
		arr_2 = np.array(img_test)
		data_2[i,:,:,:] = [arr_2[:,:,0],arr_2[:,:,1],arr_2[:,:,2]]
		label_2[i] = int(imgs_test[i].split('.')[0])
		# arr_2 = np.asarray(img_test,dtype="float32")
		# data_2[i,:,:,:] = arr_2
		# label_2[i] = int(imgs_test[i].split('.')[0])

		# 調整label

		if label_2[i] == 3:
			label_2[i] = 0
		else:
			label_2[i] = 1
	
	showPicNumber("訓練資料夾",label_1)
	showPicNumber("測試資料夾",label_2)

	return (data_1,label_1), (data_2,label_2)


load_data()





# def load_data():
# 	data_train = np.empty((49000,3,32,32),dtype="uint8") # for train
# 	label_train = np.empty((49000,),dtype="uint8")
# 	data_test = np.empty((1000,3,32,32),dtype="uint8") # for test
# 	label_test = np.empty((1000,),dtype="uint8")
	
# 	imgs_1 = os.listdir("./trainImg")
# 	num_1 = len(imgs_1)
# 	for i in range(num_1):
# 		img_1 = Image.open("./trainImg/"+imgs_1[i])
# 		arr_1 = np.array(img_1)
# 		data_train[i,:,:,:] = [arr_1[:,:,0],arr_1[:,:,1],arr_1[:,:,2]]
# 		label_train[i] = int(imgs_1[i].split('.')[0])
		
# 	imgs_2 = os.listdir("./testImg")
# 	num_2 = len(imgs_2)
# 	for i in range(num_2):
# 		img_2 = Image.open("./testImg/"+imgs_2[i])
# 		arr_2 = np.array(img_2)
# 		data_test[i,:,:,:] = [arr_2[:,:,0],arr_2[:,:,1],arr_2[:,:,2]]
# 		label_test[i] = int(imgs_2[i].split('.')[0])

# 	return (data_train,label_train), (data_test,label_test)

