from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableBranch, RunnablePassthrough

load_dotenv()

model = ChatOpenAI()

prompt1 = PromptTemplate(
    template= "Writ e a report on the topic: \n {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Summarize the following report: \n {report}",
    input_variables=['report']
)

parser = StrOutputParser()

report_chain = RunnableSequence(prompt1, model, parser)

summary_chain = RunnableBranch(
    (lambda x: len(x.split())>300, RunnableSequence(prompt2, model, parser)),
    RunnablePassthrough()
)

chain = RunnableSequence(report_chain, summary_chain)

result = chain.invoke({"topic": "Data Scientist"})

print(result)