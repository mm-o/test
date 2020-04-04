from __future__ import print_function
import numpy as np
from scipy import misc
import imageio
# import matplotlib.pyplot as plt
import glob  # 获取文件夹内容和文件信息
import re  # 获取训练信息的标志
import keras
from keras.models import Sequential  # 序列模型
from keras.layers import Dense, Dropout, Flatten  # 层函数 密集层、丢弃层、平滑层
from keras.layers import Conv2D, MaxPooling2D  # 卷积层、最大池化层
from keras.callbacks import ModelCheckpoint  # 保存检查点

def loadImages(imgpath): # 把图片读进来，转成32位数据类型
    img = imageio.imread(imgpath)
    if img is None:
     return None
    img = img.astype('float32')
    img/=255  # 除以255进行进行归一化，加快模型训练速度
    return img
def changeDim(img):   # 进行维度变化
    img=np.expand_dims(img,axis=2)  # 展成4维，灰度图再加一维
    img=np.expand_dims(img,axis=0)  # 批量通道
    return img
def loadDataset(imgpath, pathtype):
    # 准备图像数据
    images = loadImages(imgpath[0])
    images = changeDim(images)

    for single in imgpath[1:]:
        img = loadImages(single)
        img = changeDim(img)
        images =np.append(images,img, axis = 0)   # 后续图片处理，用列表附加上去
    # 准备标记
    labels = []
    pattern = re.compile(pathtype+r"\\(\d{1,2})")  # 用正则表达式读取子文件夹的文件名称
    for single in imgpath:
        label = pattern.findall(single)
        labels.append(int(label[0]))
    return np.array(images), np.array(labels)  # 图片列表和标记列表输出回去
def createModel(num_classes):
    input_shape = (28, 28, 1)
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5),
                    activation='relu',
                    padding = 'same',
                    input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))                 
    model.add(Conv2D(16, (5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

###### 准备数据
num_classes = 11  # 类别
batch_size = 120  # 批量大小
epochs = 60  # 训练次数
# 准备训练数据
imgfiles = './Train/*/*.jpg'
imgpath = glob.glob(imgfiles)
# print(imgpath)
images, labels = loadDataset(imgpath,'Train')
labels = keras.utils.to_categorical(labels, num_classes)
print(images.shape, labels.shape)
# 准备测试数据
imgfiles2 = './Test/*/*.jpg'
imgpath2 = glob.glob(imgfiles2)
# print(imgpath)
images2, labels2 = loadDataset(imgpath2,'Test')
labels2 = keras.utils.to_categorical(labels2, num_classes)
print(images2.shape, labels2.shape)

###### 设计、训练网络
model = createModel(num_classes)
model.compile(loss=keras.losses.categorical_crossentropy,  # 损失函数
              optimizer=keras.optimizers.Adadelta(),  # 优化器
              metrics=['accuracy'])  # 判别标准，正确率
# 设置检查点
checkpointer = ModelCheckpoint(filepath='./checkpoint/weights01.hdf5', verbose=1, save_best_only=True)
model.fit(images, labels,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(images2, labels2),callbacks=[checkpointer])
# 保存训练模型
model.save('./checkpoint/model01.h5')

###### 测试模型
score = model.evaluate(images2, labels2, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
