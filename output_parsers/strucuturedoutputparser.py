from langchain_openai import ChatOpenAI
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

hf_pipeline = pipeline(
    "text-generation",
    model="Qwen/Qwen2.5-1.5B-Instruct",
    max_new_tokens=200,
    do_sample=True,
    temperature = 0.8
)

model = HuggingFacePipeline(pipeline=hf_pipeline)

# load_dotenv()

# model = ChatOpenAI()

schema = [
    ResponseSchema(
        name= "fact1",
        description= "First fact about the topic"
    ),
    ResponseSchema(
        name= "fact2",
        description= "second fact about the topic"
    ),
    ResponseSchema(
        name= "fact3",
        description= "third fact about the topic"
    )
]

parser = StructuredOutputParser.from_response_schemas(schema)
prompt = PromptTemplate(
    template="Give me 3 facts about {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain = prompt | model | parser

result = chain.invoke({"topic": "Pakistan"})

print(result)