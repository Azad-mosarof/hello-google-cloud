from deepface import DeepFace
import cv2
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Images(BaseModel):
    image1: str
    image2: str

@app.get('/')
def main():
    return {'message': 'Welcome to Render Fast Api'}

@app.post('/verify')
async def verify(images: Images):

    response1 = requests.get(str(images.image1))
    response4 = requests.get(str(images.image2))

    img1_array = cv2.imdecode(np.array(bytearray(response1.content), dtype=np.uint8), -1)
    pic4_array = cv2.imdecode(np.array(bytearray(response4.content), dtype=np.uint8), -1)

    result = DeepFace.verify(img1_array, pic4_array, model_name='VGG-Face')
    result['verified'] = bool(result['verified'])
    return result