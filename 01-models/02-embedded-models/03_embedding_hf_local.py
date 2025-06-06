from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = "Dhaka is the capital of Bangladesh"

# generate a list of random text for embedding
documents = [
    "The quick brown fox jumps over the lazy dog",
    "A journey of a thousand miles begins with a single step",
    "To be or not to be, that is the question",
    "All that glitters is not gold",
    "In the middle of difficulty lies opportunity"
]

# result = embedding.embed_query(text)
# print(str(result))

results = embedding.embed_documents(documents)
print(str(results))