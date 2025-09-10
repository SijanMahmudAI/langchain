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

# Prompt template
template = PromptTemplate(
    template="Generate 5 lines of interesting facts about {topic}",
    input_variables=["topic"],
    validate_template=True
)

# create a parser
parser = StrOutputParser()

# create a chain
chain = template | model | parser 

# invoke the chain
result = chain.invoke({"topic": "Cricket"})

# Print the results
print(result)

# Print the chain graph
chain.get_graph().print_ascii()