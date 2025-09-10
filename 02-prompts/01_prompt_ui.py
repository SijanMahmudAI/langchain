from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Create llm model
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-R1-0528", task="text-generation")

# Initialize the Hugging Face chat model
model = ChatHuggingFace(llm=llm)

# Set up Streamlit UI
st.header("Research Tool")

# Get user input
# user_input = st.text_input("Enter your Prompt")

# Get the selected paper name
paper_input = st.selectbox("Select Research Paper Name", ["Attention is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"])

# Get the selected explanation style
style_input = st.selectbox("Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraph)", "Long (detailed explanation)"])

# Template for the prompt

# template = PromptTemplate(
#     template = 
#     """
#     Please summarize the research papaer titled "{paper_input}" with the following specifications:
#     Explanation Style: {style_input}
#     Explanation Length: {length_input}

#     1. Mathematical Details:
#     - Include relevant mathematical equations if present in the paper.
#     - Explain the mathematical concepts using simple, intuitive code snippets where applicable.

#     2. Analogies:
#     - Use relatable analogies to simplify complex ideas. if certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.
#     Ensure the summary is clean, accurate, and aligned with the provided style and length.

#     """,
#     input_variables=["paper_input", "style_input", "length_input"],
#     validate_template=True
# )

# OR

# Load the prompt template from the JSON file
template = load_prompt("template.json")

# Fill the Placeholder
prompt = template.invoke({
    "paper_input": paper_input,
    "style_input": style_input,
    "length_input": length_input
})

if st.button("Summarize"):
    response = model.invoke(prompt)
    st.write(response.content)