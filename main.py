import matplotlib.pyplot as plt
import xgboost as xgb
import time
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import os
from keras.models import load_model
import tensorflow_hub as hub
import math
from PIL import Image
import zipfile
import tempfile
import requests
import time
import io
import cv2



FILE_ID ='1-EpKUQQZNDdlwf4yrao6iKdK7D_LGMsd'
MODEL_URL=f'https://docs.google.com/uc?export=download&id={FILE_ID}'
MODEL_PATH='model.h5'
DATE_COLUMN = 'date/time'

def reshapeImg(image_bytes):
    # BytesIO를 사용하여 이미지로 변환
    image = Image.open(io.BytesIO(image_bytes))
    
    # 이미지 크기를 (160, 160)으로 조정
    resized_img = cv2.resize(np.array(image), (160, 160))
    resized_img = np.expand_dims(resized_img, axis=0)  # 배치 차원을 추가하여 (1, 160, 160, 3) 형태로 변환
    
    # 이미지 데이터를 0과 1 사이의 값으로 정규화
    resized_img = resized_img / 255.0
    
    return resized_img


st.title('개와 고양이를 분류해드립니다!')

st.subheader('분류해보고 싶은 개나 고양이 사진을 올려보세요')

img_file = st.file_uploader('이미지를 업로드 하세요.', type=['png', 'jpg', 'jpeg'])

# 이미지 파일 객체에서 이미지 데이터를 읽어옴
if img_file is not None:
    image_bytes = img_file.read()
    
    if not os.path.exists(MODEL_PATH):
            with requests.get(MODEL_URL, stream=True) as r:
                r.raise_for_status()
                with open(MODEL_PATH, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192): 
                        f.write(chunk)
        
    with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
            model = tf.keras.models.load_model(MODEL_PATH)
    prediction = model.predict(reshapeImg(image_bytes))
    print(prediction)
    if prediction[0][0] > prediction[0][1]:
        st.write('예측 결과: 개')
    else:
         st.write('예측 결과 : 고양이')
else:
    st.subheader('어서 올려보세요!')










