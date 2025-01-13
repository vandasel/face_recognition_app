import face_recognition as f
import logging
import re

class FaceRecognitionLoader():
    """
    Class for loading and processing images to extract face embeddings using the `face_recognition` library.

    Attributes
    ----------
    paths : list
        A list of image file paths to process.
    """

    def __init__(self, paths: list):
        self.paths = paths
        """
        Initializes the FaceRecognitionLoader instance with image paths.

        Parameters
        ----------
        paths : list
            A list of file paths to the images to be processed.
        """
    def get_faces(self) -> dict:
        """
        Detects faces in the provided images and generates embeddings.

        This method uses the `face_recognition` library to detect face locations and compute embeddings 
        for the first face found in each image. It groups results by names extracted from the file paths.

        Returns
        -------
        dict
            A dictionary where the keys are names (extracted from image paths) and 
            the values are lists of dictionaries containing:
            - 'path': str, the file path of the image
            - 'embedding': numpy.ndarray, the face encoding (128-d feature vector)
        """
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
        """
        Executes the face detection and embedding extraction workflow.

        Returns
        -------
        dict
            A dictionary containing face embeddings and metadata, structured by name.
        """
        return self.get_faces()
