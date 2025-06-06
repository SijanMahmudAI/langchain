# import modules
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal

# Load environment variables from .env file
load_dotenv()

# initialize the model
model = ChatOpenAI(model="gpt-4o", temperature=0.0)

# Schema
class Review(TypedDict):
    summary: Annotated[str, "A brief summary of the review"]
    sentiment: Annotated[Literal["pos","neg"], "The sentiment of the review (e.g., positive, negative, neutral)"]

# initialize the model with structured output
structured_model = model.with_structured_output(Review)

# Define a prompt for the model
result = structured_model.invoke(
    """The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this."""
)

# Print the structured output
print(result)
