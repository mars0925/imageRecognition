import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split

trainPath = r"D:\result\\"#圖片路徑
# np.set_printoptions(threshold = np.inf) #印出全部陣列的元素 極耗資源

# 資料夾內照片種類以及張數
images = os.listdir(trainPath)#列出檔案夾內所有檔案名稱

labels = []

for image in images:
    label = int(image.split('.')[0])
    labels.append(label)

labelset = set(labels)

for x in labelset:
    pass
    print("標籤 ", x," ", labels.count(x))


#===============================================
# import keras

# a = [0,1]
# b = np.array(a)

# c = keras.utils.to_categorical(b)

# print(c)