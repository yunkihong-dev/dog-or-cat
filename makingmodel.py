import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno
import tensorflow as tf
import os
import keras

from google.colab import drive
drive.mount('/content/drive')


from PIL import Image
import numpy as np

# 데이터 폴더 경로 설정
data_folder = 'drive/MyDrive/dataset/'  
# 각 클래스 폴더가 포함된 상위 폴더 경로

# 이미지 파일들을 저장할 리스트
train_images = []
train_labels = []
test_images = []
test_labels = []

train_folder = os.path.join(data_folder, 'training_set')
for filename in os.listdir(train_folder):
  img_path = os.path.join(train_folder, filename)
  img = Image.open(img_path)
  img = img.resize((160, 160))  # 이미지 크기 조정 (원하는 크기로 조정 가능)
  img_array = np.array(img)  # 이미지를 배열로 변환
  train_images.append(img_array)
  if filename.split('.')[0] == 'cat' :
    train_labels.append(0)
  else:
    train_labels.append(1)


    test_folder = os.path.join(data_folder, 'test_set')
for filename in os.listdir(test_folder):
  img_path = os.path.join(test_folder, filename)
  img = Image.open(img_path)
  img = img.resize((160, 160))  # 이미지 크기 조정 (원하는 크기로 조정 가능)
  img_array = np.array(img)  # 이미지를 배열로 변환
  test_images.append(img_array)
  if filename.split('.')[0] == 'cat' :
    test_labels.append(0)
  else:
    test_labels.append(1)

    train_X = np.array(train_images)
train_y = np.array(train_labels)
test_X = np.array(test_images)
test_y = np.array(test_labels)



print(test_X.shape,test_X.shape)
print(type(test_X),type(train_X))
print(test_X)

for i in train_X:
  print(i)

  train_X = train_X[0:1000]
train_y = train_y[0:1000]
test_X = test_X[0:200]
test_y = test_y[0:200]


train_X  = train_X/255
test_X = test_X/255
plt.imshow(train_X[0],cmap="Greys")
plt.imshow(test_X[0],cmap="Greys")



batch_size = 128
num_classes = 2
epochs = 25
input_shape = (160, 160, 3)


train_y = keras.utils.to_categorical(train_y, 2)
test_y = keras.utils.to_categorical(test_y, 2)

train_y.shape


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))



model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(Conv2D(64, (2, 2), activation='relu', padding='same'))


model.add(MaxPooling2D(pool_size=(2, 2)))



model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from keras.callbacks import EarlyStopping

hist = model.fit(train_X, train_y, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(test_X, test_y))
early_stopping = EarlyStopping(monitor='val_loss', patience=3)  # 3 에포크 동안 검증 손실이 감소하지 않으면 훈련 중지


from keras.models import load_model

model.save('model.h5')  # 모델 저장

model.evaluate(test_X,test_y)

