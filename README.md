# Federating learning

This project contains a federated learning Python demonstrator. It's based on a basic [TensorFlow model for clothes classification](https://www.tensorflow.org/tutorials/keras/classification) .

## Architecture
  - model.py : contains the class "Model" that allows to implement a tensorflow model, train it, evaluate it or get its weights ..
  - server.py : Flask server that contains differents endpoints for different devices to get a model or post one but also to interact with the global model.
  - device.py : Simulates two separate devices that retrieve a model from the server, train it with their own data and send it back to the server.

  - /server_training
    - /saved : contains the global model's weights (model built by the server)
    - /deviceX/received : contains the received weights of the deviceX's model
    - /to_predict : contains images to predict sent from the dashboard.
  
  - /deviceX
    - /saved : contains the weights of the device1's model
    - /received : contains the weights of the global model get from the server
    - /data : training dataset

  - /eval : Model's accuracy during all the iterations

  - /utils 
    - utils.py : Some useful functions.
    - cleaner.py : Simple class to clean the repository .
    - format_dataset.py : Functions to preprocesss, generate and display dataset.

  - /dashboard
    - index.py : A streamlit app that allows to activate device(s), get the global model's accuracy or upload an image to predict

  - /dashboard_stats
    - /index.html : A web page where it's possible to visualize the evolution of the accuracy for the global model and three devices.


## Run

1) Run the server : _python server.py <port>_
2) Run the devices : _python device/device.py <port> <device_folder>
3) Run the dashboard : _streamlit run dashboard/index.py_ and activate devices you want to be involved in the training process.
4) Open _dashboard_stats/index.html_ if you want to visualize accuracies' evolution in real time
  

## How does it work ?
![fl (2)](https://user-images.githubusercontent.com/61874108/178231813-1b25f0e9-883b-44a6-a4f7-44e0aa8fdc65.jpg)
