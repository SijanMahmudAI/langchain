# Import necessary libraries and modules
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel

# Load environment variables from a .env file
load_dotenv()

# Define a simple function to count words in a text
def word_count(text):
    return len(text.split())

# Define the prompt template for generating a joke
prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# Initialize the OpenAI chat model
model = ChatOpenAI()

# Create a string output parser
parser = StrOutputParser()

# Create a runnable sequence for generating a joke
joke_gen_chain = RunnableSequence(prompt, model, parser)

# Create a parallel runnable chain to generate the joke and count its words
parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'word_count': RunnableLambda(word_count)
})

# Combine the joke generation and word counting into a final chain
final_chain = RunnableSequence(joke_gen_chain, parallel_chain)

# Invoke the final chain with a specific topic
result = final_chain.invoke({'topic':'AI'})

# Format the final result
final_result = """{} \n word count - {}""".format(result['joke'], result['word_count'])

# Print the final result
print(final_result)