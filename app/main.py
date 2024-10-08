from app.face_models.model_mtcnn import PytorchLoader
from app.face_models.model_face_recognition import FaceRecognitionLoader
from app.calculator import calculate_dict, get_best
import chromadb
import datetime
from app.plotter import dataset_plotter
import json
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
    loader : pytorch / face_recogniton
        model for generating embeddings from faces
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
    chroma_client = chromadb.HttpClient(host='chroma_docker',port=8000)
    THRESHOLD = np.arange(0.0,1.01,0.01)

    def __init__(self,path): 
        self.loader = PytorchLoader
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
        np.random.seed(30)
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
        
    def plot_data(self):
        data = [self.train[0],self.val[0],self.test[0]]
        dataset_plotter(data=data)

    def database_input(self):
        """
        Inputs face embeddings into a ChromaDB collection.
        The embeddings get unique ids containing person's name + count of repetitions of the same person.
        """
        face_embeddings = self.loader(paths=self.train[0]).run()
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


    def query(self,data_part):
        results = {}
        """
        Queries the ChromaDB collection with a test set.

        Returns
        -------
        list
            The results of the query.
        """
        collection = self.chroma_client.get_collection("test_collection")
        face_embeddings_query = self.loader(paths=data_part).run()
        for name,embed_list in face_embeddings_query.items():
            results[name] = []
            for embedding in embed_list:
                result = collection.query(
                    query_embeddings=embedding.tolist(),
                    n_results=1
                )
                
                results[name].append(result)
        return results        


    def metrics(self,query,threshold,type):  
        """
        Calculates metrics based on query results.

        Returns
        -------
        list
            The results of the query and metric results.
        """
        threshold_results = {}        
        result_query = query
        for t in threshold:
            TP,FP,TN,FN = 0,0,0,0
            for name,results in result_query.items():
                for obj in results:
                    if obj.get("ids",[[[]]])[0][0].rstrip(digits) == name:
                        if obj.get("distances",[[[]]])[0][0] < t:
                            TP+=1
                        else:
                            FP+=1
                    else:
                        if obj.get("distances",[[[]]])[0][0] > t:
                            TN+=1
                        else:
                            FN+=1
                
            threshold_results[f"{t:.2f}"] = calculate_dict(TP=TP,TN=TN,FP=FP,FN=FN)

        with open(str("metric_tests/"+type+str(self.loader.__name__)+"*"+datetime.datetime.now().isoformat()) + ".json",'w') as file:
            file.write(json.dumps(threshold_results,indent=4))

        return threshold_results


    def metric_flow(self):
        val_metrics = self.metrics(query=self.query(data_part=self.val[0]),threshold=self.THRESHOLD,type="val")
        best_threshold = get_best(val_metrics)
        test_metrics = self.metrics(query=self.query(data_part=self.test[0]),threshold=best_threshold,type="test")
        return val_metrics,test_metrics
    
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
        return self.metric_flow()
        
    
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    start_time = time.time()
    embedder = Embedder(path="/workspaces/face_recognition_app/dataset")
    results = embedder.run()
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Time of program: {total_time:.2f}")

print() 