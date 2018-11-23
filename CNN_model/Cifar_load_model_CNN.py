import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D

model = Sequential()
model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=(32, 32, 3), activation='relu', padding='same'))
model.add(Dropout(rate=0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dropout(rate=0.25))
model.add(Dense(1024, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(10, activation='softmax'))

try:
    model.load_weights("./cifarCnnModel.h5")
    print("success")
except:
    print("error")

def plot_image(image):
    fig=plt.gcf()
    fig.set_size_inches(2, 2)
    plt.imshow(image,cmap='binary')
    plt.show()

def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig=plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25:num=25
    for i in range(0,num):
        ax=plt.subplot(5,5,1+i)
        ax.imshow(images[idx],cmap='binary')
        title="label="+str(labels[idx])
        if len(prediction)>0:
            title+=",predict="+str(prediction[idx])
        
        ax.set_title(title,fontsize=10)
        ax.set_xticks([]);ax.set_yticks([])
        idx+=1
    plt.show()
    
img=np.array(Image.open('test.png'))
plot_image(img)
data_test = np.empty((1,3,32,32),dtype="uint8")
data_test[0,:,:,:] = [img[:,:,0],img[:,:,1],img[:,:,2]]
data_test = data_test.transpose(0, 2, 3, 1)

data_test_normalize = data_test.astype('float32') / 255.0
prediction=model.predict_classes(data_test_normalize)

label_dict={0:"airplane",1:"automobile",2:"bird",3:"cat",4:"deer",
            5:"dog",6:"frog",7:"horse",8:"ship",9:"truck"}

Predicted_Probability=model.predict(data_test_normalize)
def show_Predicted_Probability(prediction, x_img,Predicted_Probability,i):
    print('predict:',label_dict[prediction[i]])
    plt.figure(figsize=(2,2))
    plt.imshow(np.reshape(data_test[i],(32, 32,3)))
    for j in range(10):
        print(label_dict[j]+
              ' Probability:%1.9f'%(Predicted_Probability[i][j]))

show_Predicted_Probability(prediction,data_test,Predicted_Probability,0)