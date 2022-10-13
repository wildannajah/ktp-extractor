import cv2
from PIL import Image
import imutils
import numpy as np
import cv2
import json
import torch
import pandas as pd
import numpy as np

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s').eval()
yolo_path = "models/best_v2.pt"
npwp_path = "models/best_checkpoint_npwp.pt"
proto_txt = "models/deploy.prototxt"
model_face_detection = "models/res10_300x300_ssd_iter_140000.caffemodel"
net = cv2.dnn.readNetFromCaffe(proto_txt,model_face_detection)
model = torch.hub.load("yolov5","custom", path=yolo_path,source="local").eval()
model.eval()

model_npwp = torch.hub.load("yolov5","custom", path=npwp_path,source="local").eval()
model_npwp.eval()

def object_count_yolov5(frame):
    b_boxes = []
    frame = frame[:, :, ::-1]
    results = model(frame,size=640)
    result_pandas  = results.pandas().xyxy[0]
    result_dict = result_pandas.to_json(orient="split")
    result_dict = json.loads(result_dict)
    data_detections = result_dict["data"]
    for det in data_detections:
        xmin,ymin,xmax,ymax,conf,_,class_det = det
        box_det = [int(xmin),int(ymin),int(xmax),int(ymax),class_det,conf]
        if conf > 0.5:
          b_boxes.append(box_det)
        else:
          continue
    return b_boxes

def object_count_npwp(frame):
    b_boxes = []
    frame = frame[:, :, ::-1]
    results = model_npwp(frame,size=640)
    result_pandas  = results.pandas().xyxy[0]
    result_dict = result_pandas.to_json(orient="split")
    result_dict = json.loads(result_dict)
    data_detections = result_dict["data"]
    for det in data_detections:
        xmin,ymin,xmax,ymax,conf,_,class_det = det
        box_det = [int(xmin),int(ymin),int(xmax),int(ymax),class_det,conf]
        if conf > 0.5:
          b_boxes.append(box_det)
        else:
          continue
    return b_boxes

def draw_yolov5(frame,b_boxes):
    for boxes in b_boxes:
        xmin,ymin,xmax,ymax,class_det,conf = boxes
        cv2.rectangle(frame,(xmin,ymin),(xmax,ymax),(0,255,0),1)
        cv2.putText(frame,str(class_det),(xmin,ymin),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1)
    return frame

def predict(img_path):
  face_exists,ktp_exists,npwp_exists = None,None,None

  frame = cv2.imread(img_path)
  frame = imutils.resize(frame, width=300)
  b_boxes = object_count_yolov5(frame)
  bbox_npwp = object_count_npwp(frame)

  (H, W) = (None, None)

  if W is None or H is None:
    (H, W) = frame.shape[:2]

  blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),(104.0, 177.0, 123.0))
  net.setInput(blob)
  detections = net.forward()

  rects = []
  for i in range(0, detections.shape[2]):
    if detections[0, 0, i, 2] > 0.8:
      box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
      rects.append(box.astype("int"))
      (startX, startY, endX, endY) = box.astype("int")
      cv2.rectangle(frame, (startX, startY), (endX, endY),
				(0, 255, 0), 1)
  print(b_boxes)
  print(bbox_npwp)
  frame = draw_yolov5(frame,b_boxes)
  frame = draw_yolov5(frame,bbox_npwp)
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  
  cv2.imwrite("kekw.jpg",frame)
  if len(rects) == 1:
    face_exists = True
  if len(b_boxes) == 1:
    ktp_exists = True
  if len(bbox_npwp) == 1:
    npwp_exists = True
  
  return {"face_exists":face_exists,"ktp_exists":ktp_exists,"npwp_exists":npwp_exists},frame