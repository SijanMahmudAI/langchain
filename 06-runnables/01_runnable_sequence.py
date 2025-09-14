# Import necessary libraries and modules
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnableSequence
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI chat model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create a string output parser
parser = StrOutputParser()

# Define the prompt1 template
prompt1 = PromptTemplate(
    template="Write me a joke about {topic}",
    input_variables=["topic"],
    validate_template=True
)

# Define the prompt2 template
prompt2 = PromptTemplate(
    template="Explain the following joke: '{text}'",
    input_variables=["text"],
    validate_template=True
)

# Create a runnable sequence (chain) by combining prompt, model, and parser
chain = RunnableSequence(prompt1, model, parser, prompt2, model, parser)

# Invoke the chain with a specific topic
result = chain.invoke({"topic": "chickens"})

# Print the results
print(result)