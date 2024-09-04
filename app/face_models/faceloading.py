import cv2 as cv
from mtcnn import MTCNN
import matplotlib.pyplot as plt # for checking results
import re


class FaceLoader():
    def __init__(self, paths):
        self.paths = paths
        self.detector = MTCNN()
        self.images = []

                    
    def get_faces(self):
        out = {}
        for path in self.paths:
            image = cv.imread(path)
            img = cv.cvtColor(image,cv.COLOR_BGR2RGB)
            face = self.detector.detect_faces(img)
            if face:
                x,y,w,h = face[0].get('box',[])
                x,y = abs(x),abs(y)
                face = img[y:y+h,x:x+w]
                face_w = cv.resize(face,(160,160))
                group = re.search(r'\/pins_(.*)\/',path)
                if group:
                    key = group.group(1)
                    if key in out:
                        out[key].append(face_w)
                    else:
                        out[key] = [face_w]
        return out


    def run(self):
        return self.get_faces()

