import torch
from PIL import Image
import re
import logging
from facenet_pytorch import MTCNN, InceptionResnetV1

class FaceLoader():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def __init__(self, paths: list):
        self.paths = paths
        self.detector = MTCNN(post_process=True, device=self.device)
        self.embedder = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)
    
    def get_faces(self) -> dict:
        out = {}
        aligned = []
        names = []
        count = 0
        iter_size = 75
        embeddings = []

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
            if key in out:
                out[key].append(embeddings[i])
            else:
                out[key] = [embeddings[i]]

        return out
    
    def run(self):
        return self.get_faces()
