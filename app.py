from fastapi import FastAPI,UploadFile,File
from fastapi.responses import  HTMLResponse,Response,StreamingResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
import cv2
import numpy
import asyncio