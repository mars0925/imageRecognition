# -*- coding: utf-8 -*-
"""
Created on Fri Nov 23 14:07:23 2018

@author: mars0925
"""

#劃出訓練圖形
import matplotlib.pyplot as plt

def show_train_history(train_history,train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()