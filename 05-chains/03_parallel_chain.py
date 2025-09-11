# Import necessary modules and libraries
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import Runnableparallel
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI chat model
model1 = ChatOpenAI()

# Initialize the Anthropic chat model
model2 = ChatAnthropic(model_name="claude-3-7-sonnet-20250219")

# 1st prompt
prompt1 = PromptTemplate(
    template="Generate short and simple notes form the following text. \n '{text}'",
    input_variables=["text"],
    validate_template=True
)

# 2nd prompt
prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text. \n '{text}'",
    input_variables=["text"],
    validate_template=True
)

# 3rd prompt to merge outputs
prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document. \n notes -> '{notes}' and \n quiz -> '{quiz}'",
    input_variables=["notes", "quiz"],
    validate_template=True
)

# Create a parser
parser = StrOutputParser()

# Parallel chain to run two prompts and models simultaneously
parallel_chain = Runnableparallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser
    }
)

# Chain to merge the outputs from the parallel chain
merge_chain = prompt3 | model1 | parser

# Combine parallel and merge chains
chain = parallel_chain | merge_chain

# Sample input text
text = """
Support vector machines (SVMs) are a set of supervised learning methods used for classification, regression and outliers detection.

The advantages of support vector machines are:

Effective in high dimensional spaces.

Still effective in cases where number of dimensions is greater than the number of samples.

Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.

Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.

The disadvantages of support vector machines include:

If the number of features is much greater than the number of samples, avoid over-fitting in choosing Kernel functions and regularization term is crucial.

SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).

The support vector machines in scikit-learn support both dense (numpy.ndarray and convertible to that by numpy.asarray) and sparse (any scipy.sparse) sample vectors as input. However, to use an SVM to make predictions for sparse data, it must have been fit on such data. For optimal performance, use C-ordered numpy.ndarray (dense) or scipy.sparse.csr_matrix (sparse) with dtype=float64.

"""

# Invoke the chain with input text
result = chain.invoke({"text": text})

# Print the final output
print("Final Output: ", result)

# Print the chain graph
chain.get_graph().print_ascii()