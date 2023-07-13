from deepface import DeepFace
from io import BytesIO
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import requests
import numpy as np
from PIL import Image

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
    model_name = 'VGG-Face'
    model = DeepFace.build_model(model_name, model=model_name + ".h5")
    return {'message': 'Welcome to Render Fast Api'}

@app.post('/verify')
def verify(images: Images):

    response1 = requests.get(str(images.image1))
    response2 = requests.get(str(images.image2))

    img1 = Image.open(BytesIO(response1.content))
    img2 = Image.open(BytesIO(response2.content))

    img1_array = np.array(img1)
    img2_array = np.array(img2)

    result = DeepFace.verify(img1_array, img2_array, model_name='VGG-Face')
    result['verified'] = bool(result['verified'])
    return result
