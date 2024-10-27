import chromadb
import numpy as np
import gradio as gr
from face_models.model_mtcnn import FaceLoader 
from face_models.model_face_recognition import FaceRecognitionLoader
from string import digits
chroma_client = chromadb.HttpClient(host='chroma_docker',port=8000)

def embedder(input_img):
    face = FaceRecognitionLoader(image=input_img).run()
    collection = chroma_client.get_collection("test_collection")
    result = collection.query(
            query_embeddings=face,
            n_results=1
        )
    if result.get("ids") and result.get("distances"):
        if result.get("distances")[0][0] <= 0.064:
            return result.get("ids")[0][0].rstrip(digits).title()
    return "Didnt find a match"

def website():
    demo = gr.Interface(embedder, gr.Image(), outputs="textbox",title="FaceRecognition",theme=gr.themes.Soft())
    demo.launch()

def run():
    website()

run()

print()