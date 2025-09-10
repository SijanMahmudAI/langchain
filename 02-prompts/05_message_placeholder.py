from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

chat_template = ChatPromptTemplate([
    ("system", "You are a helpful customer support agent."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{topic}")
])

chat_history = []

# Read chat history from a file
with open("./02-prompts/chat_history.txt") as f:
    chat_history.extend(f.readlines())

# Invoke the chat template with the chat history
prompt = chat_template.invoke({"chat_history": chat_history, "topic": "How can I reset my password?"})

# Print the response
print(prompt)

