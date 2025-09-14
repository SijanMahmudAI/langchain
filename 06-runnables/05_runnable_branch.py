# Import necessary libraries and modules
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableParallel, RunnablePassthrough, RunnableBranch, RunnableLambda

# Load environment variables from a .env file
load_dotenv()

# Define the prompt1 template for generating a report
prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

# Define the prompt2 template for summarizing text
prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)

# Initialize the OpenAI chat model
model = ChatOpenAI()

# Create a string output parser
parser = StrOutputParser()

# Create the report generation chain
report_gen_chain = prompt1 | model | parser

# Create the branch chain to summarize if the report is too long
branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model | parser),
    RunnablePassthrough()
)

# Combine the report generation and branching into a final chain
final_chain = RunnableSequence(report_gen_chain, branch_chain)

# Print the final result for a specific topic
print(final_chain.invoke({'topic':'Russia vs Ukraine'}))
