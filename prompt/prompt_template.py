from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id = 'Qwen/QwQ-32B',
    task='text_generation',
    temperature = 0
)

model = ChatHuggingFace(llm = llm)
result = model.invoke('tell me a funny 5 line poem that includes sarcasm')

print(result.content)