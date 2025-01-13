from facenet_pytorch import MTCNN, InceptionResnetV1
import logging
from PIL import Image
import re
import torch

class PytorchLoader():
    """
    Class for loading and processing images to extract face embeddings using PyTorch-based models.

    Attributes
    ----------
    device : torch.device
        The device ('cuda:0' if available, otherwise 'cpu') on which computations are performed.
    paths : list
        A list of image file paths to process.
    detector : MTCNN
        An instance of the MTCNN model for face detection and alignment.
    embedder : InceptionResnetV1
        A pre-trained InceptionResnetV1 model for generating face embeddings.
    """
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, paths: list):
        """
        Initializes the PytorchLoader instance with image paths and models.

        Parameters
        ----------
        paths : list
            A list of file paths to the images to be processed.
        """
        self.paths = paths
        self.detector = MTCNN(post_process=True, device=self.device)
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
    
    def get_faces(self) -> dict:
        """
        Detects faces in the provided images, aligns them, and generates embeddings.

        This method processes the images in batches to handle large datasets efficiently.
        For each face detected, it stores the embedding and associated metadata in a dictionary.

        Returns
        -------
        dict
            A dictionary where the keys are names (extracted from image paths) and 
            the values are lists of dictionaries containing:
            - 'path': str, the file path of the image
            - 'embedding': torch.Tensor, the face embedding
        """
        out = {}
        aligned = []
        names = []
        count = 0
        iter_size = 75
        embeddings = []
        paths = []
        for path in self.paths:
            try:
                img = Image.open(path)
                img = img.resize((160, 160)) 
                with torch.no_grad():
                    face = self.detector.detect(img)  
            except Exception as e:
                logging.error(f"Error processing image {path}: {e}")
                continue
            if face is not None:
                try:
                    with torch.no_grad():
                        align = self.detector(img)  
                    img.close()
                    if align is not None:
                        aligned.append(align)
                        group = re.search(r'\/pins_(.*)\/', path)
                        name = group.group(1) if group else "unknown"
                        names.append(name)
                        paths.append(path)
                        count += 1
                    if count % iter_size == 0:  
                        aligned_batch = torch.stack(aligned).to(self.device)
                        with torch.no_grad():
                            batch_embeddings = self.embedder(aligned_batch).detach().cpu()
                        embeddings.append(batch_embeddings)
                        logging.info(f"images : {count}")
                        aligned = []  
                        torch.cuda.empty_cache()
                except Exception as e:
                    logging.error(f"error : {e}")
                    continue
        # If the previous iterator left some aligned leftovers 
        if aligned:
            aligned_batch = torch.stack(aligned).to(self.device)
            with torch.no_grad():
                batch_embeddings = self.embedder(aligned_batch).detach().cpu()
            embeddings.append(batch_embeddings)
            logging.info(f"last batch")
        embeddings = torch.cat(embeddings, dim=0)

        for i, key in enumerate(names):

            d = {}
            d = {
                "path" : paths[i],
                "embedding" : embeddings[i] 
            }
            
            if key in out:
                out[key].append(d)
            else:
                out[key] = [d]
        return out
    
    def run(self):
        """
        Executes the facial recognition workflow.

        This method acts as the entry point for extracting embeddings from the provided dataset.

        Returns
        -------
        dict
            A dictionary containing face embeddings and metadata, structured by name.
        """
        return self.get_faces()
