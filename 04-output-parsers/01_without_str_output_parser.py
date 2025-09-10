# Import necessary libraries and modules
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
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
    template="Write a 5 line summary on the following text. \n {text}",
    input_variables=["text"],
    validate_template=True
)

# Prompt1 input
prompt1 = template1.invoke({"topic": "Artificial Intelligence"})

# Invoke the model with the prompt1
result1 = model.invoke(prompt1)

# Prompt2 input
prompt2 = template2.invoke({"text": result1.content})

# Invoke the model with the prompt2
result2 = model.invoke(prompt2)

# Print the results
print("Summary: ", result2.content)