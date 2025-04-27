import streamlit as st
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()

st.title('Business Helper')

llm = HuggingFaceEndpoint(
    repo_id='mistralai/Mistral-7B-v0.1',
    task='text_generation'
)

template = PromptTemplate(
    template = 'I want to start a {profession} Business. Tell me about {type} strategies to make my business grow.',
    input_variables= {'profession', 'type'}
)

profession = st.text_input('Enter Profession you want to know about')
type_of_strategy = st.text_input('Enter the strategy type like marketing etc')

if st.button('Get Business Strategy'):
    prompt = template.invoke({
        'profession' : profession,
        'type' : type_of_strategy
    })


    model = ChatHuggingFace(llm = llm)


    result = model.invoke(prompt)

    st.write('Here are some strategies to grow your business')
    st.write(result.content)