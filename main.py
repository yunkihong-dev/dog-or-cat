import streamlit as st
import requests
from PIL import Image
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

MODEL_URL = 'https://docs.google.com/uc?export=download&id=1-EpKUQQZNDdlwf4yrao6iKdK7D_LGMsd'
MODEL_PATH = 'model.h5'

def reshape_img(image):
    resized_img = image.resize((64, 64))
    resized_img = np.array(resized_img) / 255.0
    resized_img = np.expand_dims(resized_img, axis=0)
    return resized_img

st.title('개와 고양이를 분류해드립니다!')
st.subheader('분류해보고 싶은 개나 고양이 사진을 업로드하세요.')

img_file = st.file_uploader('이미지를 업로드하세요.', type=['png', 'jpg', 'jpeg'])

if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if not tf.io.gfile.exists(MODEL_PATH):
        with requests.get(MODEL_URL, stream=True) as r:
            r.raise_for_status()
            with open(MODEL_PATH, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)

    with tf.keras.utils.custom_object_scope({'KerasLayer': hub.KerasLayer}):
        model = tf.keras.models.load_model(MODEL_PATH)

    resized_image = reshape_img(image)
    prediction = model.predict(resized_image)

    if prediction[0][0] < prediction[0][1]:
        st.write('예측 결과: 고양이')
    else:
         st.write('예측 결과 : 개')
else:
    st.subheader('이미지를 업로드하세요!')
