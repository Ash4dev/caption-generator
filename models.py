from tqdm import tqdm
import os
from PIL import Image
import numpy as np

import streamlit as st
import pickle

from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import load_model

# to avoid re-loading of model again & again https://docs.streamlit.io/library/advanced-features/caching


@st.cache_resource
def loadModels():

    ImgModel = Xception(include_top=False,
                        pooling='avg',
                        weights="imagenet")
    FinalModel = load_model(os.path.join("dlmodels", 'modelT_Avg_9_1.h5'))

    return ImgModel, FinalModel


@st.cache_resource
def loadTokenizer():

    with open(os.path.join("picklefiles", "trainTokenizer.pickle"), 'rb') as file:
        tokenizer = pickle.load(file)

    return tokenizer
