import os
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from PIL import Image
from keras.utils import np_utils, plot_model




trainPath = r"D:\dataset32\testdata\abnormal\\"
testPath = r"D:\dataset32\testdata\abnormal\\"

images = []
labels = []

def load_data():
    imgs_train = os.listdir(trainPath)#列出訓練資料檔案夾內所有檔案名稱
    imgs_test = os.listdir(testPath)#列出測試資料檔案夾內所有檔案名稱

    firstImage = Image.open(trainPath + imgs_train[0])#開啟第一張圖片
    pixel = firstImage.size[0]#照片尺寸大小

    #訓練集
    for fileName in imgs_train:
        img_path = trainPath + fileName#完整路徑名稱
        img = image.load_img(img_path, grayscale=True, target_size=(pixel, pixel))    
        img_array = image.img_to_array(img)#將圖片轉成陣列
        images.append(img_array)#放入list

        label = int(fileName.split('.')[0])#從檔名切出標籤
        labels.append(label)#放到list

    traindata = np.array(images)#將整個list變成陣列
    trainlabels = np.array(labels)#將整個list變成陣列

    print("訓練資料集data:",traindata.shape)
    print("訓練資料集label:",trainlabels.shape)

    images.clear() #清空list,以便複用
    labels.clear()  #清空list,以便複用

    #測試集
    for fileName in imgs_test:
        img_path = testPath + fileName#完整路徑名稱
        img = image.load_img(img_path, grayscale=True, target_size=(pixel, pixel))    
        img_array = image.img_to_array(img)#將圖片轉成陣列
        images.append(img_array)#放入list

        label = int(fileName.split('.')[0])#從檔名切出標籤
        labels.append(label)#放到list

    testndata = np.array(images)#將整個list變成陣列
    testlabels = np.array(labels)#將整個list變成陣列

    print("測試資料集data:",testndata.shape)
    print("測試資料集label:",testlabels.shape) 

    return (traindata,trainlabels), (testndata,testlabels)
