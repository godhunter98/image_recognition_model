import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
import streamlit as st
import time
import sys

st.set_page_config(page_title="Fashion MNIST Predictor", layout="wide")

# importing our model architecture and model
from model_class import f_cnn
test_model = torch.load('Fashion_mnist_model.pth',map_location=torch.device('cpu'),weights_only=False)

# importing our dataset
from data import test_data

# mapping our labels to strings
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

if 'button_clicked' not in st.session_state:
    st.session_state.button_clicked = False
if 'selected_number' not in st.session_state:
    st.session_state.selected_number = 0
if 'toast_shown' not in st.session_state:
    st.session_state.toast_shown = False

st.title("Fashion MNIST Model Prediction")
st.subheader('This model takes an image of a piece of clothing as its input and returns what is it after thinking!',divider="rainbow")

st.sidebar.subheader('We got nothing to show here right now....')
st.sidebar.text('Stay tuned for more exciting stuff coming soon!!')

st.logo('resources/White_Logo.png')

if st.button('Hit me to fire up the model'):
    st.session_state.button_clicked = True

if st.session_state.button_clicked:

    col1, col2 = st.columns(2)

    with col1:
        num_test_slider = st.slider('Select a test image (0-9999)', min_value=0, max_value=9999)

    with col2:
        num_test_input = st.number_input('Or enter a specific number:', min_value=0, max_value=9999, step=1)


    # Use the input value if it's changed, otherwise use the slider value
    num_test = num_test_input if num_test_input != 0 else num_test_slider

    st.write(f'The original image is a {labels_map[test_data[num_test][1]]}!\n')

    st.image(test_data[num_test][0].squeeze().numpy(), width=500)

    # get model prediction
    with torch.no_grad():
        pred = test_model(test_data[num_test][0].view(1,1,28,28))

    st.write('But your model 🤖 thinks it is a...\n')
    time.sleep(1)
    st.write(f'{labels_map[pred.argmax(1).item()]}!\n')


    if labels_map[pred.argmax(1).item()] == labels_map[test_data[num_test][1]]:
        st.success('Congrats, it is Correct ✅!')
    else:
        st.error('Oops, it is Wrong 😑!')

if not st.session_state.toast_shown:
    st.toast('Made with love by Harsh! ♥',icon='🤖',)
    st.session_state.toast_shown = True