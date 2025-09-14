# Import necessary libraries and modules
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Initialize the OpenAI chat model
model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Create a string output parser
parser = StrOutputParser()

# Define prompt templates for generating tweets
prompt_tweet = PromptTemplate(
    template="Generate a tweet about {topic}.", 
    input_variables=["topic"],
    validate_template=True
)

# Define prompt templates for generating LinkedIn posts
prompt_linkedin = PromptTemplate(
    template="Generate a LinkedIn post about {topic}.",
    input_variables=["topic"],
    validate_template=True
)

# Create a parallel runnable chain for generating both tweet and LinkedIn post
chain = RunnableParallel(
    {
        "tweet": RunnableSequence(prompt_tweet, model, parser),
        "linkedin": RunnableSequence(prompt_linkedin, model, parser)
    }
)

# Invoke the chain with a specific topic
result = chain.invoke({"topic": "AI in healthcare"})

# Print the results
print(result)