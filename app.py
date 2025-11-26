from fastapi import FastAPI,UploadFile,File
from fastapi.responses import  HTMLResponse,Response,StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy as np
import asyncio

app = FastAPI()

fire_model = YOLO()
smoke_model= YOLO()
helmet_model= YOLO()
accident_model= YOLO()

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
        
        for model in models :
         results =model(img,verbose=False)
         img = results[0].plot()

        success,encoded_image = cv2.imencode(".jpg",img)
        if not success:
         return Response(content=b"failed to encode image",media_type="text/plain")
    except Exception as e:
     return Response(content=str(e), media_type="text/plain")
    
@app.get("/webcam")
def webcam_feed():
    async def video_stream():
        cap = cv2.VideoCapture(0)
        while True:
            ret,frame = cap.read()
            if not ret:
                break    
            
            for model in models: #runs all 4 model on webcam frame
                results = model(frame,verbose = False)
                frame = results[0].plot()

            _, encoded = cv2.imencode(".jpg",frame)

            yield (
                b"--frame\r\n"
                b"Content-Type :image/jpeg\r\n\r\n"+
                encoded.tobytes() +
                b"\r\n"
            )
            await asyncio.sleep(0.02)

    return StreamingResponse(video_stream(),media_type="multipart/x-mixed-replace;boundary=frame")