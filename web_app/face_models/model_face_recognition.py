import face_recognition as f
import dlib


class FaceRecognitionLoader():
    def __init__(self,image):
        self.image = image
    def get_faces(self) -> dict:
        face_locations = f.face_locations(self.image,model="cnn") 
        face_encodings = f.face_encodings(self.image, face_locations,model="large")
        if face_encodings:
            out = face_encodings[0]
        return out.tolist()
    
    def run(self):
        return self.get_faces()
