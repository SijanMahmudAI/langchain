# Import necessary libraries and modules
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the language model
model = ChatOpenAI()

# Define the prompt template
prompt = PromptTemplate(
    template='Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

# Initialize the output parser
parser = StrOutputParser()

# Load the text document
loader = TextLoader('./07-document-loaders/cricket.txt', encoding='utf-8')

# Load the documents
docs = loader.load()

# Print information about the loaded documents
print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

# Create a processing chain
chain = prompt | model | parser

# Invoke the chain with the content of the first document and print the result
print(chain.invoke({'poem':docs[0].page_content}))

