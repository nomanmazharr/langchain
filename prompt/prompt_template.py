from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id = 'Qwen/QwQ-32B',
    task='text_generation',
    temperature = 0.8
)

model = ChatHuggingFace(llm = llm)

template = PromptTemplate(
    template = 'tell me a 5 line poem of {poet} that includes sarcasm',
    input_variables= ['poet']
)

prompt = template.invoke(
    {'poet':'Jaun Alia'}      
)

result = model.invoke(prompt)

print(result.content)