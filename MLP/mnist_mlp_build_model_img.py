import keras
import random

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from data2 import load_data

batch_size = 64 #一次學幾張
num_classes = 2#幾個類別
epochs = 20#學幾回

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = load_data()

# #訓練和測試資料分別有多少
print("x_train_image:", x_train.shape)
print("x_train_label:", y_train.shape)

print("x_TEST_image:", x_test.shape)
print("x_TEST_label:", y_test.shape)
print("訓練集張數",len(x_train))
print("測試集張數",len(x_test))
print("圖片的像素",x_train.shape[2])


# 資料打散
index = [i for i in range(len(x_train))]
random.shuffle(index)
x_train = x_train[index]
y_train = y_train[index]

index = [i for i in range(len(x_test))]
random.shuffle(index)
x_test = x_test[index]
y_test = y_test[index]

pixel = x_train.shape[2]#圖片的像素

x_train = x_train.reshape(len(x_train), pixel* pixel)#訓練集的筆數和圖片像素1024*1024
x_test = x_test.reshape(len(x_test), pixel* pixel)#測試集的筆數和圖片像素1024*1024
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train,num_classes)
y_test = keras.utils.to_categorical(y_test,num_classes)


model = Sequential()
model.add(Dense(128, activation='relu', input_shape=(pixel* pixel,)))
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='sigmoid'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

train_history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

import matplotlib.pyplot as plt
def show_train_history(train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

show_train_history('acc','val_acc')
show_train_history('loss','val_loss')

# Save model
try:
    model.save_weights("mnist.h5")
    print("success")
except:
    print("error")
