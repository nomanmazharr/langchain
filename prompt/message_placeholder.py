from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


template = ChatPromptTemplate(
    [
        ('system', "You are a helpful assistant"),
        MessagesPlaceholder(variable_name="history"),
        ('user', "{query}")
    ]
)


history = []

with open("refund_request.txt", "r") as f:
    history.extend(f.readlines())


# print(history)

# print(template)

prompt = template.invoke({"query": "Tell me about order details", "history": history})
print(prompt)

