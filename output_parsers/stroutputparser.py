from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

template1 = PromptTemplate(
    template="tell me about {country}",
    input_variables=["country"]
)

template2 = PromptTemplate(
    template="Write a 5 line summary on \n {text}",
    input_variables=["text"]
)

strparser = StrOutputParser()


chain = template1 | model | strparser | template2 | model | strparser


result = chain.invoke({"country": "Pakistan"})
print(result)