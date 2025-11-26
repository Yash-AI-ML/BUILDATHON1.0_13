from fastapi import FastAPI,UploadFile,File
from fastapi.responses import  HTMLResponse,Response,StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy
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

model = [fire_model,smoke_model,helmet_model,accident_model]

