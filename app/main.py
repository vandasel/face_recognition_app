import chromadb
from face_models.model_mtcnn import FaceLoader
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
    threshold = 0.7
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
        
    def get_face_embeddings(self):
        """
        Extracts face embeddings from training images using the model and FaceNet.

        Returns
        -------
        dict
            A dictionary where: keys are image labels=>names, and values are lists of embeddings.
        """
        out_dict = {}
        paths = self.train[0]
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
        collection.add(
            ids=ids,  
            embeddings=embeddings
        ) 

    def query(self):
        """
        Queries the ChromaDB collection with a test set.

        Returns
        -------
        list
            The results of the query.
        """
        collection = self.chroma_client.get_collection("test_collection")
        face_q = FaceLoader(paths=self.test[0]).run()
        face_q = [el for el in face_q.values()][0][0].astype('float32')
        face_q = np.expand_dims(face_q, axis=0)
        
        query_embedding = self.embedder.embeddings(face_q)[0]  
        
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=len(self.test[0])  
        )
        return results

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

start_time = time.perf_counter()
test = Embedder(path="/workspaces/face_recognition_app/dataset").run()
end_time = time.perf_counter()

logging.debug(f"Total execution time: {end_time - start_time:.1f} seconds")
logging.debug(f"Person found : {test}")
print()


