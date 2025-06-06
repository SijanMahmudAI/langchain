from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace


# Define the Hugging Face model name
model_name = "deepseek-ai/DeepSeek-R1-0528"

# Create a Hugging Face pipeline for text generation
pipeline = HuggingFacePipeline.from_model_id(model_id=model_name, task="text-generation")

# Initialize the Hugging Face chat model with the pipeline
model = ChatHuggingFace(llm=pipeline)

# Get prompt from the user input
prompt = input("Enter your prompt: ")

# Generate a response using the chat model
response = model.invoke(prompt)

# Print the response
print("Response from Hugging Face model:", response.content)