# Import necessary libraries and modules
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Create llm model
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1-0528", 
    task="text-generation",
)

# Initialize the Hugging Face chat model
model = ChatHuggingFace(llm=llm)

# Define a Pydantic model for structured output
class Person(BaseModel):
    name: str = Field(description="The name of the person")
    age: int = Field(ge=18, description="The age of the person")
    city: str = Field(description="The city where the person lives")

# Create a Pydantic output parser
parser = PydanticOutputParser(pydantic_object=Person)

# Create a prompt template with format instructions from the parser
template = PromptTemplate(
    template= "Generate the name, age and city of a fictional {place} person. \n {format_instructions}",
    input_variables=["place"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# # Generate the prompt
# prompt = template.invoke({"place": "indian"})
# # Get the model's response
# result = model.invoke(prompt)
# # Parse the model's response into the Pydantic model
# final_result = parser.parse(result.content)


# OR

# Create a chain by combining the template, model, and parser
chain = template | model | parser

# Invoke the chain with input1
final_result = chain.invoke({"place": "bangladeshi"})

print(final_result)