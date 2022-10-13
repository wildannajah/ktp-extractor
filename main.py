
import os
from urllib import response
import PIL
import base64
import shutil
import easyocr
import pandas as pd
import numpy as np
import json
import jwt
from PIL import Image, ImageDraw
from datetime import datetime
import io
import ktp
import npwp
import urllib.request
import cv2
import imutils
import shutil
import torch
from func_detect import object_count_npwp, object_count_yolov5
from typing import Optional


from io import BytesIO
from fastapi.responses import JSONResponse,FileResponse
from starlette.middleware.cors import CORSMiddleware
from fastapi import FastAPI,File, Depends, HTTPException, status, Request, UploadFile
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.hash import bcrypt
from tortoise import fields 
from tortoise.contrib.fastapi import register_tortoise
from tortoise.contrib.pydantic import pydantic_model_creator
from tortoise.models import Model 
from pydantic import BaseModel


app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8000",
    "http://localhost:8080"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL DETECTION

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

# AUTHENTICATE

JWT_SECRET = 'myjwtsecret'

class User(Model):
    id = fields.IntField(pk=True)
    username = fields.CharField(50, unique=True)
    password_hash = fields.CharField(128)

    def verify_password(self, password):
        return bcrypt.verify(password, self.password_hash)

User_Pydantic = pydantic_model_creator(User, name='User')
UserIn_Pydantic = pydantic_model_creator(User, name='UserIn', exclude_readonly=True)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl='token')

async def authenticate_user(username: str, password: str):
    user = await User.get(username=username)
    if not user:
        return False 
    if not user.verify_password(password):
        return False
    return user 

@app.post('/token')
async def generate_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form_data.username, form_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail='Invalid username or password'
        )

    user_obj = await User_Pydantic.from_tortoise_orm(user)

    token = jwt.encode(user_obj.dict(), JWT_SECRET)

    return {'access_token' : token, 'token_type' : 'bearer'}

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=['HS256'])
        user = await User.get(id=payload.get('id'))
    except:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail='Invalid username or password'
        )

    return await User_Pydantic.from_tortoise_orm(user)


@app.post('/users', response_model=User_Pydantic)
async def create_user(user: UserIn_Pydantic):
    user_obj = User(username=user.username, password_hash=bcrypt.hash(user.password_hash))
    await user_obj.save()
    return await User_Pydantic.from_tortoise_orm(user_obj)

@app.get('/users/me', response_model=User_Pydantic)
async def get_user(user: User_Pydantic = Depends(get_current_user)):
    return user    

register_tortoise(
    app, 
    db_url='sqlite://db.sqlite3',
    modules={'models': ['main']},
    generate_schemas=True,
    add_exception_handlers=True
)



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


class int32_encoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, np.int32):

            return int(obj)

        return json.JSONEncoder.default(self, obj)



# OCR SYSTEM

reader = easyocr.Reader(['id','en'])

def draw_boxes(image, bounds, color='red', width=2):
  with Image.open(image) as im:
      draw = ImageDraw.Draw(im)
      for bound in bounds:
          p0, p1, p2, p3 = bound[0]
          draw.line([*p0, *p1, *p2, *p3, *p0], fill=color, width=width)
      return im

@app.post("/ktp", dependencies=[Depends(get_current_user)], tags=["ktp"])
async def predict(file: UploadFile = File(...)):
    try:
        upload_dir = 'data'
        image_path = os.path.join(upload_dir,file.filename)
        
        with open(image_path,"wb") as buffer:
            shutil.copyfileobj(file.file,buffer)
        
        img = Image.open(image_path)
        gray = img.convert('L')  #conversion to gray scale 
        # bw = gray.point(lambda x: 0 if x<128 else 255, '1')  #binarization 
        # bw.save("uploaded.jpg") #save it to binary
        gray.save("uploaded.jpg") #save it  to  gray

        bounds, ktp_data = ktp.ktp_to_csv(reader,"uploaded.jpg")
        pic = draw_boxes("uploaded.jpg", bounds)
        pic.save("upload_bounds.jpg")

        with open("upload_bounds.jpg", "rb") as image_file:
            base64str = base64.b64encode(image_file.read()).decode("utf-8")
        
        ktp_data["base64"] = base64str
        print("sebelum")
        response = json.dumps(ktp_data)
        print("sebelum")
        return json.loads(response)
    except Exception:
        # print(e)
        ex = {"status":"REJECTED",
              "message": "Please reupload the image, make sure the image you upload is appropriate and not less than 400 x 300 pixels"}
        return JSONResponse(content=ex)

@app.post("/npwp", dependencies=[Depends(get_current_user)], tags=["npwp"])
async def predict(file: UploadFile = File(...)):
    try:
        upload_dir = 'data'
        image_path = os.path.join(upload_dir,file.filename)
        
        with open(image_path,"wb") as buffer:
            shutil.copyfileobj(file.file,buffer)
        
        img = Image.open(image_path)
        gray = img.convert('L')  #conversion to gray scale 
        # bw = gray.point(lambda x: 0 if x<128 else 255, '1')  #binarization 
        # bw.save("uploaded.jpg") #save it to binary
        gray.save("uploaded.jpg") #save it  to  gray

        bounds, ktp_data = npwp.ktp_to_csv(reader,"uploaded.jpg")
        pic = draw_boxes("uploaded.jpg", bounds)
        pic.save("upload_bounds.jpg")

        with open("upload_bounds.jpg", "rb") as image_file:
            base64str = base64.b64encode(image_file.read()).decode("utf-8")
        
        ktp_data["base64"] = base64str
        
        response = json.dumps(ktp_data)
        return json.loads(response)
    except Exception:
        ex = {"status":"REJECTED",
              "message": "Please reupload the image, make sure the image you upload is appropriate and not less than 400 x 300 pixels"}
        return JSONResponse(content=ex)

# URL

# URL Image
@app.get("/image")
async def image():
    return FileResponse("upload_bounds.jpg")

@app.post("/ktp_url", dependencies=[Depends(get_current_user)])
async def predict(file: UploadFile = File(...), request: Request  = str):
    try:
        upload_dir = 'data'
        image_path = os.path.join(upload_dir,file.filename)
        
        with open(image_path,"wb") as buffer:
            shutil.copyfileobj(file.file,buffer)
        
        img = Image.open(image_path)
        gray = img.convert('L')  #conversion to gray scale 
        # bw = gray.point(lambda x: 0 if x<128 else 255, '1')  #binarization 
        # bw.save("uploaded.jpg") #save it to binary
        gray.save("uploaded.jpg") #save it  to  gray

        bounds, ktp_data = ktp.ktp_to_csv(reader,"uploaded.jpg")
        pic = draw_boxes("uploaded.jpg", bounds)
        pic.save("upload_bounds.jpg")

        with open("upload_bounds.jpg", "rb") as image_file:
            base64str = base64.b64encode(image_file.read()).decode("utf-8")
        ktp_data["url"] =  request.url_for("image")
        ktp_data["base64"] = base64str
        response = json.dumps(ktp_data)
        return json.loads(response)
    except Exception:
        # print(e)
        ex = {"status":"REJECTED",
              "message": "Please reupload the image, make sure the image you upload is appropriate and not less than 400 x 300 pixels"}
        return JSONResponse(content=ex)


# @app.post("/getInformation")
# async def getInformation(info : Request):
#     req_info = await info.json()
#     return {
#         "status" : "SUCCESS",
#         "data" : req_info
#     }, req_info.get('base64')

@app.post("/base64_ktp", dependencies=[Depends(get_current_user)])
async def ktp_base64(info : Request):
    try: 
        req_info = await info.json()
        base64decode = req_info.get('base64')
        print(base64decode)
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64decode, "utf-8"))))
        img.save('decoded_image.jpg')
        image_path = "decoded_image.jpg"
        img = Image.open(image_path)
        gray = img.convert('L')  #conversion to gray scale 
        # bw = gray.point(lambda x: 0 if x<128 else 255, '1')  #binarization 
        # bw.save("uploaded.jpg") #save it to binary
        gray.save("uploaded.jpg") #save it  to  gray

        bounds, ktp_data = ktp.ktp_to_csv(reader,"uploaded.jpg")
        pic = draw_boxes("uploaded.jpg", bounds)
        pic.save("upload_bounds.jpg")
        response = json.dumps(ktp_data)
        return json.loads(response)
    except Exception:
        # print(e)
        ex = {"status":"REJECTED",
              "message": "Please reupload the image, make sure the image you upload is appropriate and not less than 400 x 300 pixels"}
        return JSONResponse(content=ex)

@app.post("/url_ktp", dependencies=[Depends(get_current_user)])
async def ktp_url(info : Request):
    try: 
        req_info = await info.json()
        url = req_info.get('url')
        print(url)
        urllib.request.urlretrieve(url, "url_image.jpg")
        image_path = "url_image.jpg"
        img = Image.open(image_path)
        gray = img.convert('L')  #conversion to gray scale 
        # bw = gray.point(lambda x: 0 if x<128 else 255, '1')  #binarization 
        # bw.save("uploaded.jpg") #save it to binary
        gray.save("uploaded.jpg") #save it  to  gray

        bounds, ktp_data = ktp.ktp_to_csv(reader,"uploaded.jpg")
        pic = draw_boxes("uploaded.jpg", bounds)
        pic.save("upload_bounds.jpg")
        response = json.dumps(ktp_data)
        return json.loads(response)
    except Exception:
        # print(e)
        ex = {"status":"REJECTED",
              "message": "Please reupload the image, make sure the image you upload is appropriate and not less than 400 x 300 pixels"}
        return JSONResponse(content=ex)

@app.post("/base64_npwp", dependencies=[Depends(get_current_user)])
async def npwp_base64(info : Request):
    try: 
        req_info = await info.json()
        base64decode = req_info.get('base64')
        print(base64decode)
        img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64decode, "utf-8"))))
        img.save('decoded_image.jpg')
        image_path = "decoded_image.jpg"
        img = Image.open(image_path)
        gray = img.convert('L')  #conversion to gray scale 
        # bw = gray.point(lambda x: 0 if x<128 else 255, '1')  #binarization 
        # bw.save("uploaded.jpg") #save it to binary
        gray.save("uploaded.jpg") #save it  to  gray

        bounds, ktp_data = npwp.ktp_to_csv(reader,"uploaded.jpg")
        pic = draw_boxes("uploaded.jpg", bounds)
        pic.save("upload_bounds.jpg")
        response = json.dumps(ktp_data)
        return json.loads(response)
    except Exception:
        # print(e)
        ex = {"status":"REJECTED",
              "message": "Please reupload the image, make sure the image you upload is appropriate and not less than 400 x 300 pixels"}
        return JSONResponse(content=ex)

@app.post("/url_npwp", dependencies=[Depends(get_current_user)])
async def npwp_url(info : Request):
    try: 
        req_info = await info.json()
        url = req_info.get('url')
        print(url)
        urllib.request.urlretrieve(url, "url_image.jpg")
        image_path = "url_image.jpg"
        img = Image.open(image_path)
        gray = img.convert('L')  #conversion to gray scale 
        # bw = gray.point(lambda x: 0 if x<128 else 255, '1')  #binarization 
        # bw.save("uploaded.jpg") #save it to binary
        gray.save("uploaded.jpg") #save it  to  gray

        bounds, ktp_data = npwp.ktp_to_csv(reader,"uploaded.jpg")
        pic = draw_boxes("uploaded.jpg", bounds)
        pic.save("upload_bounds.jpg")
        response = json.dumps(ktp_data)
        return json.loads(response)
    except Exception:
        # print(e)
        ex = {"status":"REJECTED",
              "message": "Please reupload the image, make sure the image you upload is appropriate and not less than 400 x 300 pixels"}
        return JSONResponse(content=ex)


@app.post("/encode_image", dependencies=[Depends(get_current_user)])
async def predict(file: UploadFile = File(...), request: Request  = str):
    try:
        upload_dir = 'data'
        image_path = os.path.join(upload_dir,file.filename)
        
        with open(image_path,"wb") as buffer:
            shutil.copyfileobj(file.file,buffer)

        with open(image_path, "rb") as image_file:
            base64str = base64.b64encode(image_file.read()).decode("utf-8")
        dicti = {"base64" : base64str}
        response = json.dumps(dicti)
        return json.loads(response)
    except Exception:
        # print(e)
        ex = {"status":"REJECTED",
              "message": "Please reupload the image, make sure the image you upload is appropriate and not less than 400 x 300 pixels"}
        return JSONResponse(content=ex)


@app.post("/npwp_url", dependencies=[Depends(get_current_user)])
async def predict(file: UploadFile = File(...), request: Request  = str):
    try:
        upload_dir = 'data'
        image_path = os.path.join(upload_dir,file.filename)
        
        with open(image_path,"wb") as buffer:
            shutil.copyfileobj(file.file,buffer)
        
        img = Image.open(image_path)
        gray = img.convert('L')  #conversion to gray scale 
        # bw = gray.point(lambda x: 0 if x<128 else 255, '1')  #binarization 
        # bw.save("uploaded.jpg") #save it to binary
        gray.save("uploaded.jpg") #save it  to  gray

        bounds, ktp_data = npwp.ktp_to_csv(reader,"uploaded.jpg")
        pic = draw_boxes("uploaded.jpg", bounds)
        pic.save("upload_bounds.jpg")

        with open("upload_bounds.jpg", "rb") as image_file:
            base64str = base64.b64encode(image_file.read()).decode("utf-8")
        ktp_data["url"] =  request.url_for("image")
        ktp_data["base64"] = base64str
        response = json.dumps(ktp_data)
        return json.loads(response)
    except Exception:
        # print(e)
        ex = {"status":"REJECTED",
              "message": "Please reupload the image, make sure the image you upload is appropriate and not less than 400 x 300 pixels"}
        return JSONResponse(content=ex)

    
@app.post("/base64/url/ktp", dependencies=[Depends(get_current_user)])
async def ktp_base64(info : Request):
    try: 
        req_info= await info.json()
        # url
        url = req_info.get('url')
        print(url)
        if url != None:
            urllib.request.urlretrieve(url, "url_image.jpg")
            image_path = "url_image.jpg"
        else:
            base64decode = req_info.get('base64')
            print(base64decode)
            img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64decode, "utf-8"))))
            img.save('decoded_image.jpg')
            image_path = "decoded_image.jpg"
        img = Image.open(image_path)
        gray = img.convert('L')  #conversion to gray scale 
        # bw = gray.point(lambda x: 0 if x<128 else 255, '1')  #binarization 
        # bw.save("uploaded.jpg") #save it to binary
        gray.save("uploaded.jpg") #save it  to  gray

        bounds, ktp_data = ktp.ktp_to_csv(reader,"uploaded.jpg")
        pic = draw_boxes("uploaded.jpg", bounds)
        pic.save("upload_bounds.jpg")
        response = json.dumps(ktp_data)
        return json.loads(response)
    except Exception:
        # print(e)
        ex = {"status":"REJECTED",
              "message": "Please reupload the image, make sure the image you upload is appropriate and not less than 400 x 300 pixels"}
        return JSONResponse(content=ex)    

@app.post("/base64/url/npwp", dependencies=[Depends(get_current_user)])
async def ktp_base64(info : Request):
    try: 
        req_info= await info.json()
        # url
        url = req_info.get('url')
        print(url)
        if url != None:
            urllib.request.urlretrieve(url, "url_image.jpg")
            image_path = "url_image.jpg"
        else:
            base64decode = req_info.get('base64')
            print(base64decode)
            img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64decode, "utf-8"))))
            img.save('decoded_image.jpg')
            image_path = "decoded_image.jpg"
        img = Image.open(image_path)
        gray = img.convert('L')  #conversion to gray scale 
        # bw = gray.point(lambda x: 0 if x<128 else 255, '1')  #binarization 
        # bw.save("uploaded.jpg") #save it to binary
        gray.save("uploaded.jpg") #save it  to  gray

        bounds, ktp_data = npwp.ktp_to_csv(reader,"uploaded.jpg")
        pic = draw_boxes("uploaded.jpg", bounds)
        pic.save("upload_bounds.jpg")
        response = json.dumps(ktp_data)
        return json.loads(response)
    except Exception:
        # print(e)
        ex = {"status":"REJECTED",
              "message": "Please reupload the image, make sure the image you upload is appropriate and not less than 400 x 300 pixels"}
        return JSONResponse(content=ex)


# DETECTION SYSTEM

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


class int32_encoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, np.int32):

            return int(obj)

        return json.JSONEncoder.default(self, obj)


@app.post("/detect/url", dependencies=[Depends(get_current_user)])
async def predict(file: UploadFile = File(...), request: Request  = str):
    try:
        upload_dir = 'data'
        image_path = os.path.join(upload_dir,file.filename)
        
        with open(image_path,"wb") as buffer:
            shutil.copyfileobj(file.file,buffer)
        
        face_exists,ktp_exists,npwp_exists = None,None,None
        frame = cv2.imread(image_path)
        ktp_data = imutils.resize(frame, width=300)
        b_boxes = object_count_yolov5(ktp_data)
        bbox_npwp = object_count_npwp(ktp_data)

        (H, W) = (None, None)

        if W is None or H is None:
            (H, W) = ktp_data.shape[:2]

        blob = cv2.dnn.blobFromImage(ktp_data, 1.0, (W, H),(104.0, 177.0, 123.0))
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
        
        with open(image_path, "rb") as image_file:
            base64str = base64.b64encode(image_file.read()).decode("utf-8")
        print(f"base64str : {base64str}")

        data = {"face_exists":face_exists,"ktp_exists":ktp_exists,"npwp_exists":npwp_exists}

        data["url"] =  request.url_for("image")
        data["base64"] = base64str
        data2 = json.dumps(data)
        response = json.loads(data2)
        return response
    except Exception:
        # print(e)
        ex = {"status":"REJECTED",
              "message": "Please reupload the image, make sure the image you upload is appropriate and not less than 400 x 300 pixels"}
        return JSONResponse(content=ex)

   

@app.post("/detect/base64/url", dependencies=[Depends(get_current_user)])
async def ktp_base64(info : Request):
    try: 
        req_info= await info.json()
        # url
        url = req_info.get('url')
        print(url)
        if url != None:
            urllib.request.urlretrieve(url, "url_image.jpg")
            image_path = "url_image.jpg"
        else:
            base64decode = req_info.get('base64')
            print(base64decode)
            img = Image.open(io.BytesIO(base64.decodebytes(bytes(base64decode, "utf-8"))))
            img.save('decoded_image.jpg')
            image_path = "decoded_image.jpg"
        
        face_exists,ktp_exists,npwp_exists = None,None,None
        frame = cv2.imread(image_path)
        ktp_data = imutils.resize(frame, width=300)
        b_boxes = object_count_yolov5(ktp_data)
        bbox_npwp = object_count_npwp(ktp_data)

        (H, W) = (None, None)

        if W is None or H is None:
            (H, W) = ktp_data.shape[:2]

        blob = cv2.dnn.blobFromImage(ktp_data, 1.0, (W, H),(104.0, 177.0, 123.0))
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
            
        data = {"face_exists":face_exists,"ktp_exists":ktp_exists,"npwp_exists":npwp_exists}

        data2 = json.dumps(data)
        response = json.loads(data2)
        return response
    except Exception:
        # print(e)
        ex = {"status":"REJECTED",
              "message": "Please reupload the image, make sure the image you upload is appropriate and not less than 400 x 300 pixels"}
        return JSONResponse(content=ex)

