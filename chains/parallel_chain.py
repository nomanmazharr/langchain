from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel
from langchain_core.output_parsers import StrOutputParser


load_dotenv()

model1 = ChatOpenAI()
model2 = ChatAnthropic(model_name='claude-3-7-sonnet-20250219')

def load_data(filename):
    with open(filename, "r") as f:
        text = f.read()
    return text

text = load_data("dummy.txt")
prompt1 = PromptTemplate(
    template = "Generate short questions from given text for a paper: \n {text}",
    input_variables=['text']
)

prompt2 = PromptTemplate(
    template = "Generate short mcqs like options from given text for a paper: \n {text}",
    input_variables=['text']
)

parser = StrOutputParser()


parallel_chain = RunnableParallel(
    {"notes": prompt1 | model2 | parser,
    "mcqs": prompt2 | model1 | parser
})


prompt3 = PromptTemplate(
    template='Merge the notes and mcqs into a single document so I can give them as a paper to my class: \n Notes: {notes} \n MCQs: {mcqs}',
    input_variables=['notes', 'mcqs']
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain
result = chain.invoke({'text': text})

print(result)
chain.get_graph().print_ascii()