from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# model = ChatOpenAI()



model = HuggingFacePipeline.from_model_id(
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    max_new_tokens=500,
)

parser = JsonOutputParser()
template1 = PromptTemplate(
    template="Give me top youtube channels for {topic} \n {format_instructions}",
    input_variables=["topic"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


chain = template1 | model | parser
result = chain.invoke({"topic": "Artificial Intelligence"})
print(result)