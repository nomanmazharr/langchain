from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

model = ChatOpenAI()
text_loader = TextLoader("dummy.txt", encoding="utf-8").load()

prompt = PromptTemplate(
    template="Summarize the following text:\n{text}",
    input_variables=["text"],
)

parser = StrOutputParser()


chain = prompt | model | parser

result = chain.invoke({"text": text_loader})

# print(result)
# print(type(text_loader))
# print(text_loader[0].page_content)
# print(text_loader[0].metadata)
# print(text_loader[0])