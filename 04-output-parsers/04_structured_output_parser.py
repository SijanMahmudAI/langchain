# Import necessary libraries and modules
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
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

# Create a schema for structured output parser
schema = [
    ResponseSchema(name="fact_1", description="Fact 1 about the topic"),
    ResponseSchema(name="fact_2", description="Fact 2 about the topic"),
    ResponseSchema(name="fact_3", description="Fact 3 about the topic")
]

# Create a structured output parser
parser = StructuredOutputParser.from_response_schemas(schema)

# Create a prompt template with format instructions from the parser
template = PromptTemplate(
    template= "Give me 3 facts about {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# # Create the final prompt
# prompt = template.invoke({"topic": "black hole"})

# # Invoke the model with the prompt
# result = model.invoke(prompt)

# # Parse the result using the structured output parser
# final_result = parser.parse(result.content)

# OR

# Create Chain
chain = template | model | parser

# Run the chain with the input topic
final_result = chain.invoke({"topic": "black hole"})

# Print the final structured result
print(final_result)