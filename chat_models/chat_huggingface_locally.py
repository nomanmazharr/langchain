from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

os.environ['HF_HOME'] = 'D:/huggingface'

llm = HuggingFacePipeline.from_model_id(
    model_id = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task = 'text-generation'
)


model = ChatHuggingFace(llm = llm)

result = model.invoke('Tell me about cricket')

print(result.content)