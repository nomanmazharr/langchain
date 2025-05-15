from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model = ChatOpenAI()
parser = StrOutputParser()

prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the report in 5 lines \n {report}",
    input_variables=['report']
)

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Pakistan IT industry'})

print(result)
chain.get_graph().print_ascii()