from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
# openaiapi = os.getenv("OPENAI_API_KEY")

model = ChatOpenAI(
    # openai_api_key=openaiapi
)


messages = [
    SystemMessage(content="you are a helpful assistant"),
    HumanMessage(content="Tell me about famous sport in Pakistan")
]

result = model.invoke(messages)
# print(result)
messages.append(AIMessage(content=result.content))
print(messages)