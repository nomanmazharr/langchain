from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()


model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=1
)

result = model.invoke("write a 5 line poem of Jaun Elia")

print(result.content)