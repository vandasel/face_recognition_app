import cv2 as cv
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
import re
import torch

class FaceLoader():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def __init__(self, image):
        self.image = image
        self.detector = MTCNN(keep_all=True, device=self.device, post_process=False)  
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)    
    def face_embbed(self):
        face = self.detector.detect(self.image)
        if face[0] is not None and face[0][0].tolist():
            face = self.detector.detect(self.image)
            if face[0] is not None and face[0][0].tolist():
                aligned = self.detector(self.image)
                with torch.no_grad():  
                    aligned = aligned.to(self.device)  
                    face_e = self.embedder(aligned).detach().cpu()

        return face_e.tolist()[0]

    def run(self):
        return self.face_embbed()