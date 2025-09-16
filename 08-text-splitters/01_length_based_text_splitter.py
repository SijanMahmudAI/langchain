# Import necessary libraries and modules
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Initialize the PDF loader with the path to the PDF file
loader = PyPDFLoader("08-text-splitters/dl-curriculum.pdf")

# Load the document
data = loader.load()

# Initialize the text splitter
text_splitter = CharacterTextSplitter(
    chunk_size=50, 
    chunk_overlap=0, 
    separator=""
)

# Split the documents into chunks
result = text_splitter.split_documents(data) # use split_text() for plain text

# Print the resulting chunks
for i, chunk in enumerate(result):
    print(f"Chunk {i+1}:\n{chunk}\n")
