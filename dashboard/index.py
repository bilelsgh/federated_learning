from flask import request_tearing_down
from matplotlib import image
import streamlit as st
import requests
from ast import literal_eval
import sys


try:
    port = sys.argv[1]

except:
    exit("Please respect the following format: streamlit run index.py <port>")

SERVER = f"http://localhost:{port}"

st.title("Federated Learning")

img = st.file_uploader("Upload an image to predict")

with st.sidebar:
    st.title("Configuration")

    # Get global model's accuracy
    if st.button("Evaluation"):
        with st.spinner("Loading.."):
            r_acc = requests.get(f"{SERVER}/get_acc/model")
            acc = r_acc.json()["response"]
        st.code(f"Accuracy: {round(acc,2)}")
    
    # Select device to train
    d1 = st.checkbox('Device 1')
    d2 = st.checkbox('Device 2')
    d3 = st.checkbox('Device 3')

    if d1 :
        r_d = requests.post(f"{SERVER}/authorize/device1")
    if d2 :
        r_d = requests.post(f"{SERVER}/authorize/device2")
    if d3 :
        r_d = requests.post(f"{SERVER}/authorize/device3")



# Send image to predict
if img:
    st.image(img)
    with st.spinner("Sending the image.."):
        data = {"imageFile":img}
        r = requests.post(f"{SERVER}/predict",files=data)
        prediction = r.json()["pred"]

    st.success(f"The image that you sent is {prediction}")


st.markdown("""---""")

# Display active device(s)
col1, col2, col3 = st.columns(3)

with col1:
    if d1:
        st.image("dashboard/data/device_on.png", width=180)
    else:
        st.image("dashboard/data/device_off.png", width=180)
with col2:
    if d2:
        st.image("dashboard/data/device_on.png", width=180)
    else:
        st.image("dashboard/data/device_off.png", width=180)
with col3:
    if d3:
        st.image("dashboard/data/device_on.png", width=180)
    else:
        st.image("dashboard/data/device_off.png", width=180)

