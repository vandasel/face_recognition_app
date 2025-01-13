# Face recognition app

Application for facial recognition. Built with gradio. 

## Documentation

[chromadb](https://docs.trychroma.com/)

[facenet_pytorch](https://github.com/timesler/facenet-pytorch)

[face_recognition](https://github.com/ageitgey/face_recognition/)

[dataset](https://www.kaggle.com/datasets/hereisburak/pins-face-recognition/data)

[gradio](https://www.gradio.app/)

## Features

- Dockerized database
- Pre-trained models for face recognition 
- Web app built with gradio


## Requirements
```
Python==3.11.0
```

To run this application, make sure you have the following Python libraries installed. You can use the `requirements.txt` file to install them:

```plaintext
chromadb==0.5.15
opencv-python==4.10.0.84
facenet-pytorch==2.5.3
tensorflow[and-cuda]==2.17.0
scipy==1.14.1
gradio==5.3.0
face-recognition==1.3.0
seaborn==0.13.2
matplotlib==3.9.2
pandas==2.2.3
openpyxl==3.1.5
scikit-learn==1.5.2

# Additional dependencies
numpy==1.26.4
Pillow==10.4.0
protobuf==4.25.5
typing-extensions==4.12.2
h5py==3.12.1
flatbuffers==24.3.25
```
To install all required libraries, run:


```bash
pip install -r requirements.txt
```

## File Setup
**Project Structure:**  
FACE_RECOGNITION_APP/  
├── app/  
│   ├── **face_models/**  
│   │   ├── `model_mtcnn.py` - Uses PyTorch and MTCNN for face detection and embedding extraction.  
│   │   ├── `model_face_recognition.py` - Leverages the `face_recognition` library for face detection and encoding.  
│   ├── `calculator/` - Implements utilities for calculating thresholds and best matches for face embeddings.  
│   ├── `plotter/` - Contains plotting utilities for visualizing datasets, distance distributions, and classification mistakes.  
├── chroma/ - Files related to ChromaDB, used for storing and querying face embeddings.  
├── dataset/ - Directory for training, testing, and validation datasets.  
├── dlib/ - Files for dlib-related configurations or utilities.  
├── falsenegative/ - Stores false negative images detected during testing/validation.  
├── falsepositive/ - Stores false positive images detected during testing/validation.  
├── metric_tests/ - Contains performance evaluation results (e.g., `.xlsx` reports).  
├── plots/ - Stores visualizations such as confusion matrices and distance plots.  
├── web_app/ - Files and scripts for the web-based interface of the application.  
├── README.md - Documentation and guide for the project.  
├── requirements.txt - List of dependencies required for running the application.  
├── Dockerfile - Configuration for containerizing the application using Docker.  





## Docker Setup
Make sure you have Docker installed on you system.

Navigate to:
```bash
cd chroma
```
then
```bash
docker-compose up -d
```
also check if database is running
```bash
docker ps
```

## Running code
1. Visual Studio Code: Install VS Code.
2. Dev Containers Extension: Install the "Dev Containers" extension for VS Code
3. Press Ctrl+Shift+P (or Cmd+Shift+P on macOS).
4. Search for Dev Containers: Reopen in Container and select it.
5. Wait for the Container to Build.

## Results

### Validation Dataset

| **Method**             | **Confidence Threshold** | **Accuracy** | **Precision** | **Recall** | **F1**   | **Balanced Accuracy** |
|-------------------------|--------------------------|--------------|---------------|------------|----------|------------------------|
| Face Recognition Cos    | 0.937                   | 0.9589       | 0.9589        | 1.0000     | 0.9790   | 0.5068                 |
| Facenet Pytorch Cos     | 0.667                   | 0.9937       | 0.9937        | 1.0000     | 0.9968   | 0.7105                 |
| Face Recognition L2     | 0.795                   | 0.9595       | 0.9610        | 0.9982     | 0.9793   | 0.5396                 |
| Facenet Pytorch L2      | 0.552                   | 0.9920       | 0.9954        | 0.9965     | 0.9960   | 0.7760                 |

---

### Test Dataset

| **Method**             | **Confidence Threshold** | **Accuracy** | **Precision** | **Recall** | **F1**   | **Balanced Accuracy** |
|-------------------------|--------------------------|--------------|---------------|------------|----------|------------------------|
| Face Recognition Cos    | 0.937                   | 0.956        | 0.956         | 1.0000     | 0.9775   | 0.5187                 |
| Facenet Pytorch Cos     | 0.667                   | 0.9909       | 0.9908        | 1.0000     | 0.9954   | 0.6000                 |
| Face Recognition L2     | 0.795                   | 0.9560       | 0.9592        | 0.9964     | 0.9774   | 0.5371                 |
| Facenet Pytorch L2      | 0.552                   | 0.9892       | 0.9925        | 0.9965     | 0.9945   | 0.6887                 |


### Web app
![Alt text](C:/Users/kubav/Downloads/poprawna1.png)
