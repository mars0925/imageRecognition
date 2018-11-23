
# 1.資料讀取
# 2.決定模型架構(決定隱藏層層數與其中神經元數量、激活函 數、Dropout比例）。 
# 3.編譯模型(Compile model，決定模型的loss function、 optimizer、metrics)
# 4.開始訓練(Fit model)
# 5.測試結果(Evaluate)

import keras
from keras.datasets import mnist#載入資料        
num_classes = 10

# 資料讀取 
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# reshape 28*28 image to 784 floats
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# normalize
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# 步驟二 決定模型架構
# 宣告這是一個Sequential 循序性的深度學習模型 

from keras.models import Sequential
from keras.layers import Dense, Dropout
model = Sequential()
# 加入第一層hidden layer(512 neurons) # 因為第一層hidden layer需連接input vector故需要在此指定input_shape,  activation function 
model.add(Dense(512, activation='relu', input_shape=(784,)))
# 指定dropout比例
model.add(Dropout(0.2))
# 指定第二層模型hidden layer(512 neurons)、activation function、dropout 比例
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
# 指定輸出層模型 
model.add(Dense(num_classes, activation='softmax'))

model.summary()

# 步驟三、編譯模型
# 指定loss function, optimizier, metrics
from keras.optimizers import RMSprop
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

# 步驟四、開始訓練  
# 指定batch_size, nb_epoch, validation 後，開始訓練模型

batch_size = 128
epochs = 20
train_history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
# 步驟五、測試結果
# 測試結果
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
