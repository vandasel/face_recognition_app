import chromadb
from face_models.model_mtcnn import FaceLoader
from keras_facenet import FaceNet
import numpy as np
import os
from string import digits
import time 


class Embedder():
    embedder = FaceNet()
    chroma_client = chromadb.HttpClient(host='chroma_docker',port=8000)
    threshold = 0.7
    def __init__(self,path):
        self.path = path
        self.path_list = []

    def get_paths(self): 
        for dirpath, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                self.path_list.append(os.path.join(dirpath, filename))
        return self.path_list

    def get_face_embeddings(self):
        out_dict = {}
        paths = self.path_list[0:5]
        paths.extend(self.path_list[300:305])
        face_dict = FaceLoader(paths=paths).run()
        print()
        for name,faces in face_dict.items():
            embedded = []
            for face in faces:
                face_image = face.astype('float32')
                face_image = np.expand_dims(face_image,axis=0)
                out_image = self.embedder.embeddings(face_image)
                embedded.append(out_image[0])
            out_dict[name] = embedded
        return out_dict
    
    def database_input(self):
        face_embeddings = self.get_face_embeddings()
        ids = []
        embeddings = []
        self.chroma_client.delete_collection("test_collection")
        collection = self.chroma_client.create_collection("test_collection")
        for name,emb in face_embeddings.items():
            k=0
            for el in emb:
                k+=1
                key = name+str(k)
                ids.append(key)
                embeddings.append(el.tolist())
        print()
        collection.add(
            ids=ids,  
            embeddings=embeddings
        ) 

    def query(self):
        collection = self.chroma_client.get_collection("test_collection")
        
        face_q = FaceLoader(paths=[self.path_list[306]]).run()
        face_q = [el for el in face_q.values()][0][0].astype('float32')
        face_q = np.expand_dims(face_q, axis=0)
        
        query_embedding = self.embedder.embeddings(face_q)[0]  
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=1  
        )
        print()
        if results.get("distances",[[]])[0][0] <= self.threshold:
            return results.get("ids",[[]])[0][0].rstrip(digits)
        return "No match"

    def run(self):
        self.get_paths()
        self.database_input()
        return self.query()

start_time = time.perf_counter()
test = Embedder(path="/workspaces/face_recognition_app/dataset").run()
end_time = time.perf_counter()

print(f"Total execution time: {end_time - start_time:.1f} seconds")
print(f"Person found : {test}")
print()


