from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings( model="text-embedding-3-small", dimensions=32)

# Example usage of the embeddings
result = embeddings.embed_query("Hello world") # Output: [0.123, -0.456, ...] (example output, actual values will vary)

print(str(result))
