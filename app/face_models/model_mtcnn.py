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
        self.detector = MTCNN(margin=0, min_face_size=20,
                            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                            device=self.device)
        
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
        self.images = []
    def get_faces(self) -> dict:
        count = 0
        out = {}
        aligned = []
        names = []
        for path in self.paths:
            img = Image.open(path)
            face = self.detector.detect(img)
            if face is not None:
                align = self.detector(img)
                if align is not None:
                    aligned.append(align)
                    group = re.search(r'\/pins_(.*)\/', path)
                    if group:
                        name = group.group(1)
                    names.append(name)
                    count+=1
                    img.close()
                    if count % 20 == 0:
                        torch.cuda.empty_cache()
        if aligned:
            aligned = torch.stack(aligned).to(device=self.device)
            with torch.no_grad():  
                embeddings = self.embedder(aligned).detach().cpu()
            for i,key in enumerate(names):
                if key in out:
                    out[key].append(embeddings[i])
                else:
                    out[key] = [embeddings[i]]
        return out

    def run(self):
        return self.get_faces()
