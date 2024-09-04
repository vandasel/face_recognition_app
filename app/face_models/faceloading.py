import cv2 as cv
from mtcnn import MTCNN
import matplotlib.pyplot as plt # for checking results
import os
import re

class FaceLoader():
    def __init__(self, path : str):
        self.path = path
        self.detector = MTCNN()
        self.images = []
        
    def get_paths(self):
        paths = []
        for folder in os.listdir(self.path):
            folders = os.listdir(self.path +"/" + folder)
            for image in folders:
                path = self.path + "/" + folder+"/"+image
                paths.append(path)
        return paths
                    
    def get_faces(self):
        out = {}
        for path in self.get_paths():
            image = cv.imread(path)
            img = cv.cvtColor(image,cv.COLOR_BGR2RGB)
            x,y,w,h = self.detector.detect_faces(img)[0].get('box',[])
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


print()