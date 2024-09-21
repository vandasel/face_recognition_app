import torch
from PIL import Image
import re
import logging
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

class FaceLoader():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    def __init__(self, paths:list):
        self.paths = paths
        self.detector = MTCNN(image_size=160, device=self.device)  
        self.embedder = InceptionResnetV1(pretrained='casia-webface').eval()
        self.images = []
    def get_faces(self) -> dict:
        count = 0
        out = {}
        for path in self.paths:
            img = Image.open(path)
            face = self.detector.detect(img)
            if face is not None:
                aligned = self.detector(img)
                img.close()
                if aligned is not None:
                    aligned = aligned.unsqueeze(0)
                    with torch.no_grad():
                        embedding = self.embedder(aligned) 
                    count += 1
                    group = re.search(r'\/pins_(.*)\/', path)
                    if group:
                        key = group.group(1)
                        if key in out:
                            out[key].append(embedding)
                        else:
                            out[key] = [embedding]
            if count % 10 == 0:
                torch.cuda.empty_cache()
                logging.info(f"image {count}")
        return out

    def run(self):
        return self.get_faces()
