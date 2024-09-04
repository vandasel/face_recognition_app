import chromadb
from chromadb.utils import embedding_functions
from face_models.faceloading import FaceLoader
import os
import time 
 
class Embedder():
    def __init__(self,path):
        self.path = path
        self.path_list = []

    def get_paths(self): 
        for dirpath, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                self.path_list.append(os.path.join(dirpath, filename))

        return self.path_list

    def get_faces(self):
        return FaceLoader(paths=self.path_list[0:100]).run()


    def run(self):
        self.get_paths()
        return self.get_faces()
    
start_time = time.perf_counter()
test = Embedder(path="/workspaces/face_recognition_app/dataset").run()
end_time = time.perf_counter()

print(f"Total execution time: {end_time - start_time:.4f} seconds")
print()


