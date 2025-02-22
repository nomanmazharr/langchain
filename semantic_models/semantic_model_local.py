import torch
from langchain_huggingface import HuggingFaceEmbeddings
import os

os.environ['HF_HOME'] = 'F:/huggingface_cache'

llm = HuggingFaceEmbeddings(
    model_name='minishlab/M2V_multilingual_output',
    model_kwargs = {'device': 'cpu'},
    encode_kwargs = {'normalize_embeddings': True}
    )

result = llm.embed_query('Hey! how you doing')

print(result)