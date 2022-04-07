## Getting Started
To get started, install the proper dependencies either via Anaconda or Pip. I recommend Anaconda route for people using a GPU as it configures CUDA toolkit version for you.

### Conda (Recommended)

```bash
# Tensorflow CPU
conda env create -f conda-cpu.yml
conda activate yolov4-cpu

# Tensorflow GPU
conda env create -f conda-gpu.yml
conda activate yolov4-gpu
```

### Pip
(TensorFlow 2 packages require a pip version >19.0.)
```bash
# TensorFlow CPU
pip install -r requirements.txt

# TensorFlow GPU
pip install -r requirements-gpu.txt
```
### Nvidia Driver (For GPU, if you are not using Conda Environment and haven't set up CUDA yet)
Make sure to use CUDA Toolkit version 10.1 as it is the proper version for the TensorFlow version used in this repository.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

## Downloading Official YOLOv4 Pre-trained Weights
Download pre-trained yolov4.weights file: https://drive.google.com/file/d/1Q4Hvk8ur0CVCXsn3iKJ1TPlZliMVKCEY/view?usp=sharing


Copy and paste **yolov4.weights** from your downloads folder into the **'data'** folder of this repository. Make sure to have the exact same name

## Convert Yolov4 weights into tensorflow model

```bash
# Convert darknet weights to tensorflow model
python save_model.py --model yolov4 
```

## Running the Classifier with Yolov4 using Endpoints
Navigate to the main directory and run the following command 

```bash
uvicorn action_classifier_endpoint:app 
```
Running the application might take some while to load the model and expose the endpoints. You should be able to see something similar to

```bash
←[32mINFO←[0m:     Started server process [←[36m8500←[0m]
←[32mINFO←[0m:     Waiting for application startup.
←[32mINFO←[0m:     Application startup complete.
←[32mINFO←[0m:     Uvicorn running on ←[1mhttp://127.0.0.1:8000←
```

Navigate to 

```bash
http://localhost:8000/docs
```

You have set of endpoints:

1. Upload Videos
2. Check the videos that are uploaded (unclassified)
3. Select an uploaded video to be classified
4. Download the video that was classified by the model

### References  

   Huge shoutout goes to hunglc007 and nwojke for creating the backbones of this repository:
  * [AI Guy](https://github.com/theAIGuysCodee)
  * [tensorflow-yolov4-tflite](https://github.com/hunglc007/tensorflow-yolov4-tflite)
