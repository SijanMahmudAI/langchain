from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

# embedding = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# generate a list of strings about 5 indian crickerets
documents = [
    "Sachin Tendulkar is widely regarded as one of the greatest batsmen in the history of cricket.",
    "Virat Kohli is known for his aggressive batting style and has numerous records to his name.",
    "MS Dhoni, the former captain of the team and he is famous for his calm demeanor and finishing abilities in tight matches.",
    "Rohit Sharma holds the record for the highest individual score in One Day Internationals (ODIs).",
    "Kapil Dev led India to its first World Cup victory in 1983, changing the face of Indian cricket."
]

query = "Who is the captain of the Indian cricket team?"

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index,score = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[0]

print(f"Query: {query}")
print(f"Most similar document: {documents[index]}")
print(f"Similarity score: {score:.4f}")

