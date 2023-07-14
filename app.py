from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import base64
import requests
import face_recognition
import numpy as np
from PIL import Image
from io import BytesIO

def verifyFace(img1, img2, threshold=0.5):
    if img1.startswith('http'):
        img1 = Image.open(BytesIO(requests.get(img1).content))
    elif img1.startswith('data:image'):  
        img1 = Image.open(BytesIO(base64.b64decode(img1.split(',')[-1])))
    else:
        img1 = Image.open(img1)

    if img2.startswith('http'):
        img2 = Image.open(BytesIO(requests.get(img2).content))
    elif img2.startswith('data:image'):  
        img2 = Image.open(BytesIO(base64.b64decode(img2.split(',')[-1])))
    else:
        img2 = Image.open(img2)

    face_encoding1 = face_recognition.face_encodings(np.array(img1))[0]
    face_encoding2 = face_recognition.face_encodings(np.array(img2))[0]
    face_distance = face_recognition.face_distance([face_encoding1], face_encoding2)[0]

    if face_distance <= threshold:
        return {
            "distance" : face_distance,
            "verified" : True
        }
    else:
        return {
            "distance" : face_distance,
            "verified" : False
        }

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
def verify(images: Images):
    result = verifyFace(images.image1, images.image2)
    return result