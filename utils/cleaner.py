import sys
import os

def clean_devices():
    for i in range(1,4):
        for folder in ["received","saved"]:
            for file in os.listdir(f"device/device{i}/{folder}"):
                if file != ".gitkeep":
                    os.remove(f"device/device{i}/{folder}/{file}")

def clean_whole_server():
        for folder in ["device2/saved","device1/saved","device3/saved","device2/received","device1/received","device3/received","saved"]:
            for file in os.listdir(f"server_training/{folder}"):
                if file != ".gitkeep":
                    os.remove(f"server_training/{folder}/{file}")

def clean_datasets():
    for folder in ["device/device2/data","device/device1/data","device/device3/data"]:
            for file in os.listdir(f"{folder}"):
                if file != ".gitkeep":
                    os.remove(f"{folder}/{file}")

def clean_all():
    clean_devices()
    clean_whole_server()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        exit("Please call a function among these ones : clean_devices, clean_whole_server, clean_all")

    userArg = sys.argv[1]

    if userArg == "clean_devices":
        clean_devices()
    elif userArg == "clean_whole_server":
        clean_whole_server()
    elif userArg == "clean_datasets":
        clean_datasets()
    elif userArg == "clean_all":
        clean_all()
    else:
        exit("This function doesn't exist. Please chose one of these functions :\n - clean devices\n - clean_whole_server\n - clean_datasets\n - clean_all")

