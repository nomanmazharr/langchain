# from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import PydanticOutputParser
# from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field
from transformers import pipeline

# load_dotenv()

# model = ChatOpenAI()

hf_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens=200,
    do_sample=True,
    temperature = 0.8
)

model = HuggingFacePipeline(pipeline=hf_pipeline)

class Actor(BaseModel):
    name: str = Field(description="Name of the actor")
    age: int = Field(description="Age of the actor")
    sex: str = Field(description='sex of the actor')

parser = PydanticOutputParser(pydantic_object=Actor)

prompt = PromptTemplate(
    template="Give me the name age and sex from series {series_name} of any actor \n {format_instructions}",
    input_variables = ["series_name"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser

result = chain.invoke({"series_name":"Vikings"})

print(result)