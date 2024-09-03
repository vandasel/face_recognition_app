import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings

chroma_client = chromadb.HttpClient(host='chroma_docker',port=8000)

collection = chroma_client.create_collection(name="test_collection")
collection.add(
    documents=["doc1", "doc2", "doc3"],
    embeddings=[[1.1, 2.3, 3.2], [4.5, 6.9, 4.4], [1.1, 2.3, 3.2]],
    ids=['id1','id2','id3']
)

result = collection.query(
    query_embeddings=[[1.1, 2.3, 3.2]],
    n_results=3,
)
print(result.get('distances',[]))



print()


