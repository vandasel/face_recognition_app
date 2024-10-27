from app.face_models.model_mtcnn import PytorchLoader
from app.face_models.model_face_recognition import FaceRecognitionLoader
from app.calculator import get_best
import chromadb
import datetime
from app.plotter import dataset_plotter,plot_distances, save_mistakes
import json
import logging
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from PIL import Image
from string import digits
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
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
    THRESHOLD = np.arange(0.000,1.001,0.001)

    def __init__(self,path,dist_calc): 
        self.loader = FaceRecognitionLoader
        self.path = path
        self.dist_calc = dist_calc
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
        n = 3000
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
        paths = []

        self.chroma_client.delete_collection("test_collection")
        collection = self.chroma_client.create_collection(
            name = "test_collection",
            metadata = {"hnsw:space": self.dist_calc}
            )
        
        for name,emb in face_embeddings.items():
            k=0
            for el in emb:
                k+=1
                key = name+str(k)
                ids.append(key)
                embeddings.append(el.get("embedding").tolist())
                paths.append(el.get("path"))

        collection.add(
            documents=paths,  
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

        for name,dictt in face_embeddings_query.items():
            results[name] = []
            for item in dictt:
                result = collection.query(
                    query_embeddings=item.get("embedding").tolist(),
                    n_results=1
                )
                
                results[name].append((result,item.get("path")))
        return results        





    def metrics(self, query, threshold, model):
        """
        Calculates metrics based on query results and collects distances for visualization.
        Now also calculates classification metrics using scikit-learn.
        """
        threshold_results = {}
        values = []
        correctness = []  
        result_query = query
        

        for t in threshold:
            k = 0
            y_true = [] 
            y_pred = [] 
            for name, results in result_query.items():
                for obj in results:
                    distance = obj[0].get("distances", [[[]]])[0][0]
                    distance = 1/(1+distance)
                    values.append(distance)       
                    ground_truth = name    
                    predicted_label = obj[0].get("ids", [[[]]])[0][0].rstrip(digits)
                    
                    if predicted_label == ground_truth:  
                        if distance >= t:
                            y_true.append(0)  
                            y_pred.append(0)  
                            correctness.append("right")
                        else:
                            y_true.append(1)  
                            y_pred.append(0)  
                            correctness.append("wrong")
                            if model == "test":
                                save_mistakes(obj=obj, k=k, filedir=f"falsenegative/{self.loader.__name__}/{self.dist_calc}/")
                    else:
                        if distance >= t:
                            y_true.append(0)  
                            y_pred.append(1)  
                            correctness.append("wrong")
                            if model == "test":
                                save_mistakes(obj=obj, k=k, filedir=f"falsepositive/{self.loader.__name__}/{self.dist_calc}/")
                        else:
                            y_true.append(1)  
                            y_pred.append(1)  
                            correctness.append("right")
                    k += 1
            accuracy = accuracy_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
   
            

            threshold_results[f"{t:.3f}"]={
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }

        if model == "test":
            plot_distances(values, correctness, model,self.loader.__name__,self.dist_calc)
            cm = confusion_matrix(y_true, y_pred)
            disp = ConfusionMatrixDisplay(cm).plot(cmap='plasma')
            plt.grid(False)
            disp.ax_.set_xticklabels(['P', 'N'])
            disp.ax_.set_yticklabels(['P', 'N'])
            plt.savefig(f"plots/threshold_{t:.3f}_{self.loader.__name__}_{self.dist_calc}_{model}_cm.png")
            plt.close()
        
        df = pd.DataFrame(threshold_results).T
        excel_filename = f"metric_tests/{model}_{self.loader.__name__}_{self.dist_calc}_{datetime.datetime.now().isoformat()}.xlsx"
        df.to_excel(excel_filename, index=True)

        return threshold_results



    def metric_flow(self):
        val_metrics = self.metrics(query=self.query(data_part=self.test[0]),threshold=self.THRESHOLD,model="val")
        best_threshold = get_best(val_metrics)
        test_metrics = self.metrics(query=self.query(data_part=self.test[0]),threshold=best_threshold,model="test")
        return val_metrics, test_metrics
    
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
        self.plot_data()
        self.database_input()
        return self.metric_flow()
        
    
        
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    start_time = time.time()
    embedder = Embedder(path="/workspaces/face_recognition_app/dataset", dist_calc="cosine")
    results = embedder.run()
    end_time = time.time()
    total_time = end_time - start_time
    logging.info(f"Time of program: {total_time:.2f}")

print()