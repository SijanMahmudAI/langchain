# Import necessary libraries and modules
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create llm model
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1-0528", task="text-generation")

# Initialize the Hugging Face chat model
model = ChatHuggingFace(llm=llm)

# 1st prompt
template1 = PromptTemplate(
    template="Write a detailed report on {topic}",
    input_variables=["topic"],
    validate_template=True
)

# 2nd prompt
template2 = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"],
    validate_template=True
)

# create a parser
parser = StrOutputParser()

# create a chain
chain = template1 | model | parser | template2 | model | parser

# invoke the chain
result = chain.invoke({"topic": "Artificial Intelligence"})

# Print the results
print("Summary: ", result)

# Print the chain graph
chain.get_graph().print_ascii()