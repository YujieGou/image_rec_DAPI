# -*- coding: utf-8 -*-
"""
Created on 06-12-2019
@author: Gou Yujie
"""
import cv2
import numpy as np
from keras.models import Sequential
from sklearn.utils.multiclass import type_of_target
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from sklearn import metrics
import pylab as plt
from keras.regularizers import l2
import re
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  #强制选择cpu版本的TensorFlow
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
w=h=56
c=3
#读入图片
def get_data(path):
    data=[]
    file=[]
    for root,dirs,files in os.walk(path):
        for filename in files:
            file_path=os.path.join(root,filename)
            cut_fold = file_path.split('\\')[4]
            tag_fold=file_path.split('\\')[5]
     #       print(tag)
            if cut_fold == 'merge_cut':
                img = cv2.imread(file_path)
                file.append(img)
                ret1 = re.search("pos_cut_[1-9]", tag_fold)
                ret2=re.match("neg_cut_[1-9]", tag_fold)
                if ret1:
                    data.append('1')
                elif ret2:
                    data.append('0')
                else:
                    print(file_path)
            else:
                   print('wrong')
    file = np.array(file, dtype="float")
    data=np.array(data,dtype="float")
#    data_pos = np.array(data_pos, dtype="float")
    return data,file

label,file=get_data('F:\cuckoo_lab\image_project\pics\merge_cut\\')  #分成训练集和测试集
#print(label.shape,file.shape)
#x,y=get_data("F:\cuckoo_lab\image_project\pics\merge_cut\\")
#print(x.shape,y.shape,label.shape,file.shape)
#用函数分组
x_train, x_valid, y_train, y_valid = train_test_split(file, label, test_size=0.3, random_state=3)
_, x_test, _, y_test = train_test_split(file, label, test_size=0.4, random_state=3)
#处理数据格式，并不知道为啥这样处理但不处理跑不动
x_train=x_train.reshape(x_train.shape[0],c,w,h)
x_train=x_train.astype('float32')/255.0
x_valid=x_valid.reshape(x_valid.shape[0],c,w,h)
x_valid=x_valid.astype('float32')/255.0
#print(x_train.shape,x_valid.shape)
x_test=x_test.reshape(x_test.shape[0],c,w,h)  #（226,3,56,56）
x_test=x_test.astype('float32')/255.0  #np.array()
y_train = to_categorical(y_train, 2)
y_valid = to_categorical(y_valid, 2)
y_test = to_categorical(y_test, 2) #(226,2)
#print(type_of_target(y_test))
#print(y_test)
lb = LabelBinarizer()
#labels = lb.fit_transform(label)  # transfer label to binary value
#labels = to_categorical(label)
#print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
#print(type(x_test),x_valid,type(y_test),y_valid,type(x_test))
#构建卷积神经网络#自己写的 尚可
def cnn_model(train_data,train_label,test_data,test_label):
    model = Sequential()

#卷积层
    model.add(Conv2D(
        filters = 16,
        kernel_size = (3,3),
        padding = "valid",
        data_format = "channels_first",
        input_shape = (c,w,h),
        activation='relu')
    )
    # model.add(Activation('relu'))#激活函数使用修正线性单元
#池化层
    model.add(MaxPooling2D(
        pool_size = (2,2),
        strides = (2,2),
        padding = "valid"))
#卷积层
    model.add(Conv2D(
        32,
        (3,3),
        padding = "valid",
        data_format = "channels_first",
        activation='relu')
    )
    # model.add(Activation('relu'))
# 池化层
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid"))

#卷积层
    model.add(Conv2D(
        64,
        (3,3),
        padding = "valid",
        data_format = "channels_first"))
    model.add(Activation('relu'))
# 池化层
    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding="valid"))
#卷积层
    model.add(Conv2D(
        128,
        (3,3),
        padding = "valid",
        data_format = "channels_first"))
    model.add(Activation('relu'))


    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))

    model.add(Activation('relu'))
    
    model.add(Dense(20,kernel_regularizer=l2(0.003)))

    model.add(Dropout(0.4))

    model.add(Dense(2,activation='softmax'))
    #model.add(Activation('softmax'))
    adam = Adam(lr = 0.001)
    model.compile(optimizer = adam,
            loss =  'categorical_crossentropy',
            metrics = ['accuracy'])
    print ('----------------training-----------------------')
    model.fit(train_data,train_label,batch_size=400,epochs = 80)
    print ('----------------testing------------------------')
    loss,accuracy = model.evaluate(test_data,test_label)
    print ('\n test loss:',loss)
    print ('\n test accuracy',accuracy)

   # score=model.predict_proba(x_test)
    classes=model.predict_proba(x_test).astype('float32')[:,1]
   # classes = np_utils.to_categorical(classes, 2)
 #   cla=LabelBinarizer(cla, classes=[0, 1])
#    classes = np.array(cla)
#    classes = [np.argmax(one_hot) for one_hot in classes]
#    print(type_of_target(classes))
    return classes

#学弟之前写的
def buildModel(x_train):
    model = Sequential()
    model.add(Conv2D(
        input_shape=( x_train.shape[1],x_train.shape[2], x_train.shape[3]),
        filters=32,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        activation='relu',
        data_format="channels_first")
    )

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding='valid',
        activation='relu',
        data_format="channels_first")
    )

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
                  )

    model.summary()
    model.fit(x_train, y_train, batch_size=400, epochs=15)
    model.evaluate(x_valid, y_valid)
    classes = model.predict_classes(x_test).astype('float32')
    classes = np.array(classes)
    return classes

#画出auc图
def plot(pred):
    fpr, tpr, threshold = metrics.roc_curve(y_test, pred,pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(6, 6))
    plt.title('my model ROC')
    plt.plot(fpr, tpr, 'b', label='Val AUC = %0.3f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig("roc_predict1.png")
    plt.show()

classes=cnn_model(x_train,y_train,x_valid,y_valid)
#classes=buildModel(x_train)
y_test = [np.argmax(one_hot) for one_hot in y_test] #将独热标签转回一维数组才能画图
y_test=np.array(y_test)
#print(type_of_target(y_test))
plot(classes)


