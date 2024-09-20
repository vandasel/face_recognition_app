import cv2 as cv
from facenet_pytorch import MTCNN
import matplotlib.pyplot as plt # for checking results
import re
import torch
import logging

class FaceLoader():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def __init__(self, paths:list):
        self.paths = paths
        self.detector = MTCNN(keep_all=True, device=self.device, post_process=False)
        self.images = []

                    
    def get_faces(self) -> dict:
        count=0
        out = {}
        for path in self.paths:
            image = cv.imread(path)
            img = cv.cvtColor(image,cv.COLOR_BGR2RGB)
            face = self.detector.detect(img)
            if face[0] is not None and face[0][0].tolist():
                count+=1
                x,y,w,h = face[0][0]
                x,y = abs(x),abs(y)
                x,y,w,h = int(x),int(y),int(w),int(h)
                face = img[x:x+w,y:y+h]
                face_w = cv.resize(face,(160,160))
                group = re.search(r'\/pins_(.*)\/',path)
                logging.info(f"image {count}")
                if group:
                    key = group.group(1)
                    if key in out:
                        out[key].append(face_w)
                    else:
                        out[key] = [face_w]
        return out


    def run(self):
        return self.get_faces()