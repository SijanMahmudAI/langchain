# Import necessary libraries and modules
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough
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

# Create a runnable sequence for generating a joke
joke_gen_chain = RunnableSequence(prompt1, model, parser)

# Create a parallel runnable chain to generate the joke and its explanation
parallel_chain = RunnableParallel(
    {
        "joke": RunnablePassthrough(),
        "joke_explanation": RunnableSequence(prompt2, model, parser)
    }
)

# Combine the joke generation and explanation into a final chain
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Invoke the final chain with a specific topic
result = final_chain.invoke({"topic": "Artificial Intelligence"})

# Print the results
print(result)