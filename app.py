from fastapi import FastAPI,UploadFile,File
from fastapi.responses import  HTMLResponse,Response,StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import asyncio

app = FastAPI()

fire_model = YOLO(),
smoke_model= YOLO(),
helmet_model= YOLO(),
accident_model= YOLO(),

fire_model.overrides["classes"] =[0]
smoke_model.overrides["classes"] =[0]
helmet_model.overrides["classes"] =[0,1]
accident_model.overrides["classes"] =[0]

models= [fire_model,smoke_model,helmet_model,accident_model]

app.mount("/static",StaticFiles(directory="static"),name = "static")


@app.get("/",response_class=HTMLResponse)
def home_page():
    with open("static/index.html") as f:
         return HTMLResponse(f.read())

@app.post("/predict-image")
async def predict_image(file:UploadFile=File(...)):


    try:
        image_bytes = await file.read()
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return Response(contentb="invalid image file",media_type="text/plain")
        
    for model in models:
        results = model(img,verbose=False)
        img = results[0].plot


        