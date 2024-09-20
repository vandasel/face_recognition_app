from app.calculator import calculate_dict, get_best
import chromadb
import datetime
from app.face_models.model_mtcnn import FaceLoader
import json
from keras_facenet import FaceNet
import logging
import numpy as np
import os
from string import digits
import time 


class Embedder():
    """
    Class used for facial recognition algorithm TBC.

    Attributes
    ----------
    embedder : FaceNet
        FaceNet model for generating embeddings from faces
    chroma_client : chromadb.HttpClient
        A ChromaDB server client, connected by docker network
    threshold : for now float, than list
        A threshold used for face match results
    path : str
        Path to the dataset's directory
    path_list : list
        File paths for all images in the directory
    train : list
        List of paths for training images
    test : list
        List of paths for testing images
    val : list
        List of paths for validation images
    """
    embedder = FaceNet()
    chroma_client = chromadb.HttpClient(host='chroma_docker',port=8000)
    THRESHOLD = np.arange(0.0,1.85,0.05)

    def __init__(self,path):
        self.path = path
        self.path_list = []
        self.train = []
        self.test = []
        self.val = []

        """
        Initializes variables for future use as self.

        Parameters
        ----------
        path : str
            Path to the dataset's directory
        """

    def get_paths(self): 
        """
        Collects the paths of all images in the dataset

        Returns
        -------
        list
            A shuffled list of file paths
        """
        for dirpath, dirnames, filenames in os.walk(self.path):
            for filename in filenames:
                self.path_list.append(os.path.join(dirpath, filename))

        np.random.shuffle(self.path_list)
        return self.path_list

    def split_data(self):
        """
        Splits the dataset
        The split is 80% training, 10% testing, and 10% validation.
        """
        n = len(self.path_list)
        self.train.append(self.path_list[:int(0.8 * n)])
        self.test.append(self.path_list[int(0.8 * n):int(0.8 * n)+int(0.1 * n)])
        self.val.append(self.path_list[int(0.8 * n)+int(0.1 * n):])
        
    def get_face_embeddings(self,paths):
        """
        Extracts face embeddings from training images using the model and FaceNet.

        Returns
        -------
        dict
            A dictionary where: keys are image labels=>names, and values are lists of embeddings.
        """
        out_dict = {}
        face_dict = FaceLoader(paths=paths).run()
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
        """
        Inputs face embeddings into a ChromaDB collection.
        The embeddings get unique ids containing person's name + count of repetitions of the same person.
        """
        face_embeddings = self.get_face_embeddings(paths=self.path_list)
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
        collection.add(
            documents=ids,  
            embeddings=embeddings,
            ids = ids
        ) 


    def query(self):
        results = {}
        threshold_results = {}
        """
        Queries the ChromaDB collection with a test set.

        Returns
        -------
        list
            The results of the query.
        """
        collection = self.chroma_client.get_collection("test_collection")
        face_embeddings_query = self.get_face_embeddings(paths=self.val[0])
        for threshold in list(self.THRESHOLD):
            TP,TN,FP,FN = 0,0,0,0
            for name,embed_list in face_embeddings_query.items():
                for embedding in embed_list:
                    result = collection.query(
                        query_embeddings=embedding.tolist(),
                        n_results=1
                    )
                    if name in results:
                        results[name.rstrip(digits)].append(result.get("distances")[0][0])
                    else:
                        results[name.rstrip(digits)] = [result.get("distances")[0][0]]

                    if result.get("ids",[[[]]])[0][0].rstrip(digits) == name:
                        if result.get("distances",[[[]]])[0][0] < threshold:
                            TP+=1
                        else:
                            FP+=1
                    else:
                        if result.get("distances",[[[]]])[0][0] > threshold:
                            TN+=1
                        else:
                            FN+=1

            threshold_results[f"{threshold:.2f}"] = calculate_dict(TP=TP,TN=TN,FP=FP,FN=FN)
        with open(str(datetime.datetime.now().isoformat()) + ".txt",'w') as file:
            file.write(json.dumps(threshold_results,indent=4))

        return results, get_best(threshold_results)

    def run(self):
        """
        Executes the entire workflow in order.

        Returns
        -------
        list
            The results of the face recognition alg.
        """
        self.get_paths()
        self.split_data()
        self.database_input()
        return self.query()

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    start_time = time.time()
    embedder = Embedder(path="/workspaces/face_recognition_app/dataset")
    results = embedder.run()
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Time of program: {total_time:.2f}")


print()