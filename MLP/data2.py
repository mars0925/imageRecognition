
#coding:utf-8
#照片放在同一個資料夾
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

trainPath = r"D:\result\\"#圖片路徑
# np.set_printoptions(threshold = np.inf) #印出全部陣列的元素 極耗資源

#讀取資料夾mnist下的42000張圖片，圖片為灰階圖，所以為1通道，圖像大小28*28


def load_data():
	imgs_1 = os.listdir(trainPath)#列出檔案夾內所有檔案名稱
	firstImage = Image.open(trainPath + imgs_1[0])#開啟第一張圖片
	pixel = firstImage.size[0]#照片尺寸大小
	num_1 = len(imgs_1)#檔案夾有多少圖片

	#產生陣列
	data_1 = np.empty((num_1,1,pixel,pixel),dtype="float32")#圖片為灰階圖，所以為1通道
	label_1 = np.empty((num_1,),dtype="uint8")
	
	for i in range(num_1):
		img_1 = Image.open(trainPath + imgs_1[i])
		arr_1 = np.asarray(img_1,dtype="float32")
		data_1[i,:,:,:] = arr_1
		original = int(imgs_1[i].split('.')[0])
		# #調整label
		if original == 2:
			label_1[i] = 0
		else:
			label_1[i] = 1



	# 隨機抽取20%的測試集
	trainData, testData, trainLabel, testLabel = train_test_split(data_1, label_1, test_size=0.2)	
	
	# 資料夾內照片種類以及張數
	labelset = set(label_1)
	labels = list(label_1)

	for x in labelset:
	    pass
	    print("標籤 ", x," ", labels.count(x))

	return (trainData,trainLabel), (testData,testLabel)