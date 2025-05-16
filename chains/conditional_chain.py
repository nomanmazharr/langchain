from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch, RunnableLambda

load_dotenv()
class Feedback(BaseModel):
    sentiment: Literal['positive', 'negative'] = Field(description="Sentiment of the feedback")

parser1 = PydanticOutputParser(pydantic_object=Feedback)
parser2 = StrOutputParser()

model = ChatOpenAI()

prompt1 = PromptTemplate(
    template="Analyze the sentiment of the following feedback: {feedback} \n {format_instructions}",
    input_variables=['feedback'],
    partial_variables={'format_instructions':parser1.get_format_instructions()}
)


classifier_feedback_chain = prompt1 | model | parser1

prompt2 = PromptTemplate(
    template = "Write an appropriate response to the positive feedback: \n {feedback}",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template="Write an appropriate response to the negative feedback: \n {feedback}",
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x: x.sentiment=='positive', prompt2 | model | parser2),
    (lambda x: x.sentiment=='negative', prompt3 | model | parser2),
    RunnableLambda(lambda x: "could not find sentiment")
)


chain = classifier_feedback_chain | branch_chain

result = chain.invoke({'feedback': "I like this watch"})

print(result)