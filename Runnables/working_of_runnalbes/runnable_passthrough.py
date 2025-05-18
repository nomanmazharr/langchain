from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnablePassthrough, RunnableParallel

load_dotenv()

model = ChatOpenAI()

prompt1 = PromptTemplate(
    template= "tell me a joke about the {person}",
    input_variables=['person']
)

prompt2 = PromptTemplate(
    template="Explain the following joke : \n {joke}",
    input_variables=['joke']
)

parser = StrOutputParser()

joke_gen = RunnableSequence(prompt1, model, parser)

parallel_chain = RunnableParallel({
    'joke': RunnablePassthrough(),
    'explanation': RunnableSequence(prompt2, model, parser)
})

chain = RunnableSequence(joke_gen, parallel_chain)

result = chain.invoke({'person': 'Data Scientist'})
# print(result)
print(result['joke'])
print(result['explanation'])