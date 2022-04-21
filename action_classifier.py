import os
from paths import video_classified_dir, video_received_dir

# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import time
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
import core.utils as utils
from core.yolov4 import filter_boxes
from core.functions import *
from tensorflow.python.saved_model import tag_constants
from PIL import Image
import cv2
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


class Process:
    def __init__(self, path, model, videoId):
        response = requests.get(api_url + 'videos/{}'.format(path))
        self.video_url = response.json()['videoRawUrl']
        self.path = path
        self.model = model
        self.videoId = videoId

    def detect(self):
        # load configuration for object detector
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)
        STRIDES = np.array(cfg.YOLO.STRIDES)
        ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS, False)
        XYSCALE = cfg.YOLO.XYSCALE
        NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
        input_size = 416
        video_path = video_received_dir + self.path
        # get video name by using split method
        video_name = video_path.split('/')[-1]
        video_name = video_name.split('.')[0]

        infer = self.model.signatures['serving_default']


        vid = cv2.VideoCapture(self.video_url)

        out = None

        # by default VideoCapture returns float instead of int
        width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(vid.get(cv2.CAP_PROP_FPS))
        codec = cv2.VideoWriter_fourcc(*'H264')
        # output_path = video_classified_dir + self.path
        output_path = video_classified_dir + os.path.splitext(self.path)[0] + ".avi"
        out = cv2.VideoWriter(output_path, codec, fps, (width, height))

        frame_num = 0
        # while video is running
        while True:
            return_value, frame = vid.read()
            if return_value:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
            else:
                print('Video has ended or failed, try a different video format!')
                break
            frame_num += 1
            print('Frame #: ', frame_num)

            frame_size = frame.shape[:2]
            image_data = cv2.resize(frame, (input_size, input_size))
            image_data = image_data / 255.
            image_data = image_data[np.newaxis, ...].astype(np.float32)
            start_time = time.time()

            batch_data = tf.constant(image_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

            boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
                boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
                scores=tf.reshape(
                    pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
                max_output_size_per_class=50,
                max_total_size=50,
                iou_threshold=0.45,
                score_threshold=0.5
            )

            # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
            original_h, original_w, _ = frame.shape
            bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

            pred_bbox = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]

            # read in all class names from config
            class_names = utils.read_class_names(cfg.YOLO.CLASSES)

            # by default allow all classes in .names file
            allowed_classes = list(class_names.values())

            # custom allowed classes (uncomment line below to allow detections for only people)
            # allowed_classes = ['person']

            image = utils.draw_bbox(frame, pred_bbox, True)

            send_request(pred_bbox, allowed_classes, frame_num, self.videoId)

            fps = 1.0 / (time.time() - start_time)
            print("FPS: %.2f" % fps)
            # result = np.asarray(image)
            # cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
            result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            out.write(result)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cv2.destroyAllWindows()
