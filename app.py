import streamlit as st
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dense, Lambda, ELU, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing import image 
from extract_bottleneck_features import *
import json

dog_names=[]
with open('data/dog_names.json') as json_file:
    dog_names = json.load(json_file)

bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_Resnet = bottleneck_features['train']

Resnet_Model = Sequential()

Resnet_Model.add(GlobalAveragePooling2D(input_shape = train_Resnet.shape[1:]))
Resnet_Model.add(Dense(133,activation='softmax'))
Resnet_Model.load_weights('saved_models/weights.best.Resnet.hdf5')


st.title('Dog Breed Classifier')
st.subheader('Classification using Deep Learning')


def img_conv(imag):
    x = image.load_img(imag, target_size=(224, 224))
    x = image.img_to_array(x)
    return np.expand_dims(x, axis=0)


uploaded_file = st.file_uploader("Choose a image file", type=("jpg","png"))
if uploaded_file is not None:
    imagee = Image.open(uploaded_file)
    x = img_conv(uploaded_file)
    x = Resnet_Model.predict(extract_Resnet50(x))
    name = dog_names[np.argmax(x)].rsplit('.',1)[1].replace("_", " ")
    st.image(imagee, caption='This image looks like - '+name)

    





