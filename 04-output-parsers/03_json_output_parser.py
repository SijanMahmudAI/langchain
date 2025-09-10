# Import necessary libraries and modules
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create llm model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528", 
    task="text-generation",
    temperature=2.0,
)

# Initialize the Hugging Face chat model
model = ChatHuggingFace(llm=llm)

# Create a JSON output parser
parser = JsonOutputParser()

template = PromptTemplate(
    template= "Give me the name, age and city of a fictional person \n{format_instructions}",
    input_variables=[],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)


# Format the prompt using the template
# prompt = template.format()

# Get the result from the model
# result = model.invoke(prompt)

# Parse the result using the JSON output parser
# final_result = parser.parse(result.content)

# Or

# Create a chain
chain = template | model | parser

# Invoke the chain
final_result = chain.invoke({})

# Print the final result
print(final_result)

# Print the type of the final result
print(type(final_result))