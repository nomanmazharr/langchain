from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

UserInput = [
    SystemMessage(content="You are a helpful assistant")
]
while True:
    user_input = input("You: ")
    UserInput.append(HumanMessage(content=user_input))
    if user_input.lower() == "q":
        break
    result = model.invoke(UserInput)
    print(result.content)