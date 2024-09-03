from mtcnn import MTCNN
import os
import cv2 as cv


class FaceLoader():
    def __init__(self, path : str):
        self.path = path
        self.detector = MTCNN()

    def load_images(self):
        ...

    def run(self):
        ...    

