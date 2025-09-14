# Import necessary libraries and classes
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

# Initialize the DirectoryLoader with the specified path, file pattern, and loader class
loader = DirectoryLoader(
    path='./07-document-loaders/books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

# Load documents lazily from the specified directory
docs = loader.lazy_load()

# Print metadata of all loaded documents
for document in docs:
    print(document.metadata)