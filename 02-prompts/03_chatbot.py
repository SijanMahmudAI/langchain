from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

# Create huggingface endpoint model
llm = HuggingFaceEndpoint(repo_id="meta-llama/Llama-3.1-8B-Instruct", task="text-generation")

# Initialize the Hugging Face chat model
chat_model = ChatHuggingFace(llm=llm)

# Store the chat history
chats = [SystemMessage(
    content="You are a helpful assistant. Answer the user's questions to the best of your ability. you speak in english language."
)] 

# Loop to interact with the chatbot
while True:
    user= input("You: ")

    # Append user message to chat history
    chats.append(HumanMessage(content=user))

    # Check for exit command
    if user.lower() in ["exit", "quit", "q"]:
        print("Exiting the chatbot. Goodbye!")
        break

    # Invoke the chat model
    result = chat_model.invoke(chats)

    # Append AI response to chat history
    chats.append(AIMessage(content=result.content))

    # Print the AI response
    print("Bot: ",result.content)

print("Chat history:", chats)