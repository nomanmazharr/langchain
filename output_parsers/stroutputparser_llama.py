from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv

load_dotenv()
os.environ['HF_HOME'] = 'F:/huggingface'

llm = HuggingFacePipeline.from_model_id(
    model_id='meta-llama/Llama-3.2-1B',
    task='text-generation'
)

template1 = PromptTemplate(
    template = """You are a helpful AI historian.
        Write a full report on the {year} war between Pakistan and India.
        Include causes, major events, casualties, and consequences.
        The report should be informative and detailed.""",
    input_variables=["year"]
)

template2 = PromptTemplate(
    template = 'Summarise this report in 10 lines: \n  {text}',
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | llm | parser | template2 | llm | parser

result = chain.invoke({"year": "Feb 2019"})
print(result)