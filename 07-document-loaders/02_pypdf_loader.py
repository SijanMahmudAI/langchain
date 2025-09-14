# Import necessary libraries and modules
from langchain_community.document_loaders import PyPDFLoader

# Initialize the PDF loader with the path to the PDF file
loader = PyPDFLoader('./07-document-loaders/dl-curriculum.pdf')

# Load the documents from the PDF
docs = loader.load()

# Print the number of documents loaded
print(len(docs))

# Print the content and metadata of the loaded documents
print(docs[0].page_content)
print(docs[1].metadata)