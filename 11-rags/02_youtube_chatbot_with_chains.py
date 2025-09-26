# Import all necessary libraries and modules
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough, RunnableLambda
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()



# Initialize the chat model
model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")

# Initialize a StrOutputParser
parser = StrOutputParser()




# Get the video url here:
video_url = input("Enter the YouTube video URL: ")

# Function to extract video ID from URL
def get_video_id(url: str) -> str:
    parsed_url = urlparse(url)
    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query)["v"][0]   # normal youtube link
    elif parsed_url.hostname == "youtu.be":
        return parsed_url.path[1:]                  # short link
    else:
        raise ValueError("Invalid YouTube URL")

# Call the function with the provided URL
video_id = get_video_id(video_url)

# Fetch and print the transcript
try:
    # Load the transcript using the YouTubeTranscriptApi
    ytt_api = YouTubeTranscriptApi()
    response = ytt_api.fetch(video_id,languages=["en"])

    # Combine all text snippets into a single transcript
    transcript = " ".join([chunk.text for chunk in response.snippets])
except:
    print("No caption available for this video.")

# Initialize the text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

# Split the transcript into smaller chunks
chunks = splitter.create_documents([transcript])

# Initialize the embedding model
embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

# Create a FAISS vector store from the chunks and embeddings
vectorstore = FAISS.from_documents(chunks, embedding)




# Create a retriever from the vector store
retriever = vectorstore.as_retriever(
    search_type="similarity", 
    search_kwargs={"k": 4}
)

def format_docs(retrieved_docs):
    context ="\n\n".join([doc.page_content for doc in retrieved_docs])
    return context


# Create a Prompt
prompt = PromptTemplate(
    template=
    """
    You are a helpful assistant.
    Answer Only from the provided transcript context.
    If the context is insufficient, just say you don't know.
    Context: {context}
    Question: {question}
    """,
    input_variables=["context", "question"]
)




# Loop to interact with the chatbot
while True:
    # Get the user query
    user= input("Query: ")

    # Check for exit command
    if user.lower() in ["exit", "quit", "q"]:
        print("Exiting the chatbot. Goodbye!")
        break

    # Create a parallel chain
    parallel_chain = RunnableParallel(
        {
            "question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(format_docs)
        }
    )

    # Create the final chain
    final_chain = parallel_chain | prompt | model | parser

    # Invoke the final chain with the user query
    result = final_chain.invoke(user)

    # Print the generated response
    print(f"Answer: {result}\n\n")


