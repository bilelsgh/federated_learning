# TensorFlow and tf.keras
# from cgi import test
# from gc import callbacks
# from statistics import mode
# from tkinter import E
from gzip import READ
from lzma import MODE_FAST
from tkinter import ALL, LAST
import tensorflow as tf

# Helper libraries
from flask import Flask, send_file, jsonify, request
import os 
import sys
import numpy as np
import pandas as pd
from utils.utils import check_dir, get_labels 

from model import Model
from utils.cleaner import clean_all
from PIL import Image
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from flask_cors import CORS
import ast

app = Flask(__name__)
CORS(app)

# ---------------- INIT ---------------- 
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

save_path = "server_training" # Where the devices' weights are saved
checkpoint_path = "server_training/cp.ckpt" # Where the global model's weights are saved
checkpoint_dir = os.path.dirname(checkpoint_path)

clean_all() # Clean the server
LABELS = [i for i in range(8)] # Classes on which the global model trains
MODEL = Model("server_training",LABELS,"server_training") # Global model
MODEL.save_weights()
MODEL_perf = Model("server_training",LABELS,"server_training") # Global model with the best accuracy


DEVICE_ACC = {"MODEL_perf": [0], "EVAL_ACC": [0], "device1": [0], "device2": [0], "device3": [0]} # Accuracies throughout iterations
ALL_DEVICES = ast.literal_eval(os.getenv("DEVICES")) # Every devices involved, it can contain devices that are disconnected 
READY_DEVICES = [] # Devices that have all their weights files and that are then ready to be merged. 
CONNECTED_DEVICES = [] # Devices that are currently part of the process
GET_DEVICES= [] # Devices that have get the current global model
POST_DEVICES = [] # Devices that have post their weights during the current iteration
ITERATION = 0

# ------------------------------------------


@app.route("/",methods=['GET'])
def testGet():
    return "It seems to work !"


@app.route("/authorize/<device>", methods=['POST'])
# Add a device to ALL_DEVICES
def authorize(device):
    global ALL_DEVICES

    if device not in ALL_DEVICES:
        ALL_DEVICES.append(device)

    print(ALL_DEVICES)
    return "Success"

@app.route("/predict", methods=["GET","POST"])
# Predict an image sent from an app
def predict():
    global MODEL_perf
    #obj = {0: "Apple", 1: "Banana", 2: "Chips", 3: "Candy", 4: "Car", 5: "Pasta", 6: "Yogurt"}
    obj = {"0":"a top", "1": "a trouser", "2": "a pullover", "3": "a dress", "4": "a coat", 
                "5": "a sandal", "6": "a shirt", "7": "a sneaker", "8": "a bag", "9": "an ankle boot"}

   
    if request.method == 'POST':
        f = request.files['imageFile']
        f.save(f"server_training/to_predict/{secure_filename(f.filename)}")

        img = Image.open(f)
        np_img = np.array(img)
        print(f"# Shape = {np_img.shape}")
        pred_array = MODEL_perf.predict(np.array([np_img]))
        print(pred_array)
        pred = np.argmax(pred_array[0], axis=0)
        os.remove(f"server_training/to_predict/"+ secure_filename(f.filename))

        final_response = jsonify({'pred':  obj[pred]})
        final_response.headers.add('Access-Control-Allow-Origin', '*')
        return final_response



@app.route("/disconnect/<device>", methods=['POST'])
# Disconnect a device, we won't wait it anymore for merge but we still consider it if we have stored its weights
def disconnect(device):
    global CONNECTED_DEVICES
    try:
        CONNECTED_DEVICES.remove(device)
        if device in POST_DEVICES:
            POST_DEVICES.remove(device)
        if device in GET_DEVICES:
            GET_DEVICES.remove(device)
        final_response = jsonify({'response': f"Device <{device}> has been removed succesfully."})
        print(f"!{device} is deconnected. {CONNECTED_DEVICES}")
        return final_response, 200
    except:
        final_response = jsonify({'response': "An error has occured."})
        return final_response, 500


@app.route("/get_acc/<id>", methods=['GET'])
# Get devices' or model's accuracy
def get_acc(id):
    global DEVICE_ACC, MODEL, MODEL_perf

    if id == "model":
        final_response = jsonify({'response': MODEL_perf.eval()})
    else:
        final_response = jsonify({'response': str(DEVICE_ACC[id])})
    final_response.headers.add('Access-Control-Allow-Origin', '*')
    
    return final_response


# Get weight
@app.route("/get_model_index/<device>",methods=['GET'])
def get_model_index(device):
    global MODEL, GET_DEVICES
    
    MODEL.acLock()
    if device not in GET_DEVICES:
        print(f"({device}) got the indexes")
        return send_file("server_training/saved/cp.ckpt.index")
    
    else:
        return "You can't get the index now.", 500

@app.route("/get_model_weights/<device>",methods=['GET'])
def get_model_weights(device):
    global MODEL

    if device not in GET_DEVICES:
        print(f"({device}) got the weights")
        GET_DEVICES.append(device)
        MODEL.reLock()
        return send_file("server_training/saved/cp.ckpt.data-00000-of-00001")
    
    else:
        MODEL.reLock()
        return "You can't get the weights now.", 500



@app.route("/send_model/<info>",methods=['POST'])
# Receive devices' weights and merge them
def send_model(info):
    global DEVICE_ACC, MODEL, READY_DEVICES, ALL_DEVICES, ITERATION, LABELS, CONNECTED_DEVICES, GET_DEVICES, POST_DEVICES, MODEL_perf

    MODEL.acLock()
    READY_DEVICES = []
    print(f"\n*~*~*~ ITERATION N°{ITERATION} *~*~*~\n")

    device = info 
    
    if device not in ALL_DEVICES:
        MODEL.reLock()
        return "Not allowed", 500

    # A new device is connected
    new_device_connected = False 
    if device not in CONNECTED_DEVICES:
        CONNECTED_DEVICES.append(device)
        new_device_connected = True

    # A new device has posted its weight
    if device not in POST_DEVICES:
        print(f"({device}) added in POST")
        POST_DEVICES.append(device)

    path = os.path.join(save_path, device, "received") # server_training/<deviceX>/received
    if not os.path.isdir(path):
        os.mkdir(path)

    # Remove weights file
    for file in os.listdir(path):
        if file != ".gitkeep":
            os.remove(f"{path}/{file}")

    # Save the weights file just received and update accuracy
    files_ = ( request.files )
    DEVICE_ACC[device].append( float(request.form["accuracy"]) )

    for file in list( files_.keys() ):
        if file != 'labels.txt':
            files_[file].save(f"server_training/{device}/received/{files_[file].filename}")
        else:
            files_[file].save(f"server_training/{device}/{files_[file].filename}")


    # Which device(s) is/are ready to be merged ?
    for device_ in ALL_DEVICES:
        if check_dir(device_) and device_ not in READY_DEVICES:
            print(f"[{device_} is ready]")
            READY_DEVICES.append(device_)

    print(f"------------\n--> There is/are {len(ALL_DEVICES)} device(s) involved : {ALL_DEVICES}")
    print(f"\n--> There is/are {len(CONNECTED_DEVICES)} device(s) connected : {CONNECTED_DEVICES}")
    print(f"\n--> There is/are {len(READY_DEVICES)} device(s) ready to be merged: {READY_DEVICES}\n------------")
    print(f"\n--> There is/are {len(POST_DEVICES)} device(s) that have/has sent their weights : {POST_DEVICES}")
    print(f"\n--> There is/are {len(GET_DEVICES)} device(s) that have/has pulled the global model : {GET_DEVICES}")

    # We wait for every connected devices to merge
    are_connected_devices_ready = [device in POST_DEVICES for device in CONNECTED_DEVICES]
    print(f"Are connected : {are_connected_devices_ready}")
    if False not in are_connected_devices_ready and len(POST_DEVICES) > 1:

        models = []
        devices_labels = []
        ITERATION += 1
        
        # Create several models based on ready devices' weights just received
        print(f":Got all files! I'm about to merge {' + '.join(READY_DEVICES)} ({len(READY_DEVICES)}).\n")
        for ready_device in READY_DEVICES:
            model1 = Model(f"server_training/{ready_device}", get_labels(ready_device), "server_training")
            model1.load_model()
            models.append(model1)
            devices_labels += get_labels(ready_device)

        # Update the test dataset according to the devices' train dataset
        devices_labels = set(devices_labels)
        if LABELS != devices_labels:
            MODEL.set_labels_range(devices_labels)
            MODEL_perf.set_labels_range(devices_labels)
            LABELS = list( map( lambda x: str(x), devices_labels) )

        print(f":Test dataset contains the following classes: {', '.join(LABELS)[:-2]}")

        # Merge
        acc_before = MODEL.eval()
        print(f"\n# Before merging\n - Accuracy: {acc_before}")
        MODEL.average_model(models) # Create a new model by combining several devices' weights
 
        print(f"\n:Merge done.\n..Evaluation\n")
        print(f"\n# After merging\n")
        acc = MODEL.eval()

        # Is accuracy better than MODEL_perf ? Is a new device connected ?
        if acc > MODEL_perf.eval() or new_device_connected:
            new_device_connected = False
            MODEL_perf.perf_load_model() 

        DEVICE_ACC["MODEL_perf"].append(MODEL_perf.eval())
        DEVICE_ACC["EVAL_ACC"].append(acc)
        GET_DEVICES= []
        POST_DEVICES = []
        READY_DEVICES = []

        # Write excel
        eval_df = pd.DataFrame(DEVICE_ACC["EVAL_ACC"])
        eval_df.to_excel("eval/eval_fl.xlsx")
        print("> Iteration over.\n\n")
    
    MODEL.reLock()
    
    return "Model received"


if __name__ == "__main__":
    MODEL.create_model()
    try:
        port = sys.argv[1]
    except:
        print("Please precise the port.")
        exit()
        
    app.run(debug=False, port=port)

