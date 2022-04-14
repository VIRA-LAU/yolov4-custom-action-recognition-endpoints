import shutil
from enum import Enum
import os
from paths import video_classified_dir, video_received_dir
import firebase_admin
from firebase_admin import credentials, storage
import requests
from fastapi import FastAPI, File, UploadFile
from starlette.responses import FileResponse
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from os import walk

from action_classifier import Process

saved_model_loaded = tf.saved_model.load('./checkpoints/yolov4-416', tags=[tag_constants.SERVING])

# Firebase initialization
cred = credentials.Certificate("./data/fire.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'fyp-interface.appspot.com'})
bucket = storage.bucket()

app = FastAPI()

api_url = "https://stats-service-fyp-vira.herokuapp.com/api/v1/"


# uvicorn main:app --reload

class ModelName(str, Enum):
    yolov4 = "yolov4"


@app.get("/api/v1/public")
async def root():
    return {"message": "Working Public Endpoint"}


@app.get("/api/v1/public/models/{model_name}")
async def get_model(model_name: ModelName):
    isDarknetYOLOv4Available = os.path.isfile('./data/yolov4.weights')
    isTensorflowYOLOv4Available = os.path.isfile('./checkpoints/yolov4-416/saved_model.pb')
    if model_name == ModelName.yolov4:
        if not isDarknetYOLOv4Available:
            return {"model_name": model_name, "Darknet YOLOV4 Available": False}

        if not isTensorflowYOLOv4Available:
            return {"model_name": model_name, "Tensorflow YOLOV4 Available": False}

    return {"model_name": model_name, "Darknet YOLOV4 Available": True, "Tensorflow YOLOV4 Available": True, }


@app.post("/api/v1/public/upload-video")
async def uploadVideo(file: UploadFile = File(...)):
    received_path = './video_received/'
    if not os.path.exists(video_received_dir):
        os.mkdir(video_received_dir)

    # save the video in google drive and get a url --> https://jkaskddhaskd/video1
    # service -----> https://jkaskddhaskd/video1
    # getURlLinks() ---> {
    # ur

    # }
    with open(received_path + '{}'.format(file.filename), 'wb') as buffer:
        shutil.copyfileobj(file.file, buffer)

    return {"filenames": file.filename}


@app.get("/api/v1/public/getVideo/{path}")
def getVideo(path):
    file_path = os.path.join(video_classified_dir + path)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="video/mp4", filename="{}.mp4".format(path))
    return {"error": "File not found!"}


@app.get("/api/v1/public/classified-videos")
def getListOfClassifiedVideos():
    if not os.path.exists(video_classified_dir):
        os.mkdir(video_classified_dir)
    filenames = next(walk(video_classified_dir), (None, None, []))[2]
    return {"Classified Videos": filenames}


@app.get("/api/v1/public/received-videos")
def getListOfReceivedVideos():
    if not os.path.exists(video_received_dir):
        os.mkdir(video_received_dir)
    filenames = next(walk(video_received_dir), (None, None, []))[2]
    print(filenames)
    return {"Received Videos": filenames}


@app.get("/api/v1/public/classify-video/{path}")
def ClassifyVideo(path):
    request = {
        "videoName": path,
        "videoLocation": "LAU Court"
    }

    response = requests.get(api_url + 'videos/{}'.format(path))
    videoId = response.json()['videoId']
    result = Process(path, saved_model_loaded, videoId);
    result.detect()
    print('Classified')

    blob = bucket.blob('classified_videos/' + path)
    blob.upload_from_filename(video_classified_dir + path)
    blob.make_public()
    print("Firebase URL:", blob.public_url)

    classified_url = {
        "videoClassifyUrl": blob.public_url
    }
    requests.put(api_url + 'videos/update/{}'.format(videoId), json=classified_url)

    return {'Success'}
