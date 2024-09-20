import cv2 as cv
from facenet_pytorch import MTCNN
import numpy as np
from keras_facenet import FaceNet
import re
import torch

class FaceLoader():
    embedder = FaceNet()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def __init__(self, image):
        self.image = image
        self.detector = MTCNN(keep_all=True, device=self.device, post_process=False)      
    def face_embbed(self):
        face = self.detector.detect(self.image)
        if face[0] is not None and face[0][0].tolist():
            x,y,w,h = face[0][0]
            x,y = abs(x),abs(y)
            x,y,w,h = int(x),int(y),int(w),int(h)
            face = self.image[x:x+w,y:y+h]
            face_w = cv.resize(face,(160,160))
            face_image = face_w.astype('float32')
            face_image = np.expand_dims(face_image,axis=0)
            out_image = self.embedder.embeddings(face_image)
        return out_image

    def run(self):
        return self.face_embbed()