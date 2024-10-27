import face_recognition as f
import logging
import re

class FaceRecognitionLoader():
    def __init__(self, paths: list):
        self.paths = paths

    def get_faces(self) -> dict:
        out = {}
        count = 1
        for path in self.paths:
            try:
                image = f.load_image_file(path) 
                face_locations = f.face_locations(image, model="cnn") 
                face_encodings = f.face_encodings(image, face_locations, model="large")
                if face_encodings:
                    group = re.search(r'\/pins_(.*)\/', path)
                    name = group.group(1) if group else "unknown"
                    
                  
                    d = {
                        "path": path,
                        "embedding": face_encodings[0]  
                    }

                    if count % 75 == 0:
                        logging.info(f"face found nr: {count}")
                    
                   
                    if name not in out:
                        out[name] = []
                    out[name].append(d)

                    count += 1

            except Exception as e:
                logging.error(f"Error processing image {path}: {e}")
                continue

        return out

    def run(self):
        return self.get_faces()
