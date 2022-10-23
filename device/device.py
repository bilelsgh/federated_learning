# TensorFlow and tf.keras
from ctypes.wintypes import POINT
from distutils.log import error
from importlib.metadata import files
import tensorflow as tf

# Helper libraries
import numpy as np
import pandas as pd
import requests
import os
import sys
sys.path.append('.')
from model import Model
import time
import random
from dotenv import load_dotenv

        

if __name__ == "__main__":

    try:
        port = sys.argv[1]
        DEVICE = sys.argv[2]
        labels = sys.argv[3].split("/")
        labels = list( map( lambda x: int(x), labels ) )
    except:
        exit("Please respect the following format: pythonX device.py <port> <device_folder> <label1/labels2/...>")


    try:
        # Init
        load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))
        model = Model(f"device/{DEVICE}",labels,f"device/{DEVICE}") # save_path, classes' labels, data_path
        model.eval()
        ITER=0
        EVAL_ACC = []
        SERVER = os.getenv("SERVER")

        while(ITER <= 100):
        
            # Get the model..
            error_500 = True
            while error_500 :
                print("\n# About to get the model")
                resp = requests.get(f"{SERVER}:{port}/get_model_index/{DEVICE}")
                print(resp)
                status = [int(resp.status_code)]
                open(f"device/{DEVICE}/received/cp.ckpt.index","wb").write(resp.content)
                
                resp = requests.get(f"{SERVER}:{port}/get_model_weights/{DEVICE}")
                open(f"device/{DEVICE}/received/cp.ckpt.data-00000-of-00001","wb").write(resp.content)
                status.append(int(resp.status_code))
                print(f"Status: {status}")

                error_500 = 500 in status

                if error_500:
                    rest = 5
                    print(f"Can't get the model now, I'll try again in {rest} sec.")
                    time.sleep(5)

            # ..and load it !
            print("\n#############")
            model.load_model()
            model.eval()
            print("#############\n")

            # Train with new data
            model.train_model()
            print("\n# Train with new data")
            acc = model.eval()

            # Send weights and labels
            print("\nFile(s) sent:")
            files_  = {}
            status = 500
            while status == 500:
                for idx,file in enumerate(os.listdir(f"device/{DEVICE}/saved")[2:]):
                    file_ = f"{file}_0" if idx != 0 else f"{file}_1"
                    files_[file] = open(f'device/{DEVICE}/saved/{file}', "rb")
                print("About to send")      
                r = requests.post(f'{SERVER}:{port}/send_model/{DEVICE}', files=files_, data={"accuracy": float(acc)} )
                status = r.status_code

                for k in list( files_.keys() ) :
                    files_[k].close()
                
                time.sleep(6)


            # Rest
            rest = random.randint(1,10)
            print(f" >> Step : {ITER}/100, I'll just sleep during {rest} sec ..")
            time.sleep(1)
            ITER += 1

        
            # Write excel
            EVAL_ACC.append(acc)
            eval_df = pd.DataFrame(EVAL_ACC)
            eval_df.to_excel(f"eval/eval_{DEVICE}_fl.xlsx")
            

    except KeyboardInterrupt:
        print("..This device is disconnecting...")
        r = requests.post(f'{SERVER}:{port}/disconnect/{DEVICE}')
        print("Disconnected.")

