import chromadb
import numpy as np
import gradio as gr
from face_models.model_mtcnn import FaceLoader 
from face_models.model_face_recognition import FaceRecognitionLoader
from string import digits
import time as t

chroma_client = chromadb.HttpClient(host='chroma_docker',port=8000)

def embedder(input_img):

    modelstart = t.time()
    face = FaceRecognitionLoader(image=input_img).run()
    modelstop = t.time()

    dbstart = t.time()
    collection = chroma_client.get_collection("test_collection")
    result = collection.query(
            query_embeddings=face,
            n_results=1
        )
    dbstop = t.time()
    if result.get("ids") and result.get("distances"):
        conf = 1/(result.get("distances")[0][0] + 1)
        if conf >= 0.954:
            return result.get("ids")[0][0].rstrip(digits).title(), modelstop-modelstart, dbstop-dbstart
    return "Didnt find a match", modelstop-modelstart, dbstop-dbstart

def website():
    demo = gr.Interface(
    fn=embedder, 
    inputs=gr.Image(),  
    outputs=[
        gr.Textbox(label="Rozpoznana osoba"), 
        gr.Number(label="Czas przetwarzania modelu (s)"), 
        gr.Number(label="Czas przetwarzania bazy danych (s)"),  
    ],
    title="System rozpoznawania twarzy",
    theme=gr.themes.Monochrome()
)
    demo.launch()

def run():
    website()

run()

print()