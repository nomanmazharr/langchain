from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel


load_dotenv()

model = ChatOpenAI()

prompt1 = PromptTemplate(
    template="Generate me a tweet about the {topic}",
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template="Generate a linkedin post about the {topic}",
    input_varialbes=['topic']
)


parser = StrOutputParser()

chain = RunnableParallel({
    'tweet': RunnableSequence(prompt1, model, parser),
    'post': RunnableSequence(prompt2, model, parser)
})

result = chain.invoke({"topic": "Model Context Protocol Servers"})

# print(result)
print(result['tweet'])
print(result['post'])