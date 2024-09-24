import face_recognition as f
import logging
import dlib
import re

class FaceLoader2():
    def __init__(self,paths:list):
        self.paths = paths
    def get_faces(self) -> dict:
        out = {}
        count = 1
        for path in self.paths:
            image = f.load_image_file(path) 
            face_locations = f.face_locations(image,model="cnn") 
            face_encodings = f.face_encodings(image, face_locations)
            if face_encodings:
                group = re.search(r'\/pins_(.*)\/', path)
                name = group.group(1) if group else "unknown"
                if count % 75 == 0:
                    logging.info(f"face found nr: {count}")
                if name not in out:
                    out[name] = [] 
                out[name].append(face_encodings[0]) 
                count+=1
        return out
    
    def run(self):
        return self.get_faces()
