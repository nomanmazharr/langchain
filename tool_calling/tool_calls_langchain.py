from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

@tool
def multiply(a:int, b:int) -> int:
  """Multiply two numbers"""
  return a*b

# print(multiply.invoke({'a':6, 'b':8}))

llm1 = ChatAnthropic(model='claude-3-5-sonnet-latest')

# print(llm1.invoke('hi'))

llm_with_tools = llm1.bind_tools([multiply]) # we can add any number of tools we have in the list

print(llm_with_tools) # llm_with_tools now contain the tools we have given to LLM when need we can give query and it will execute the tool that is required 

# This just shows how we bind tools in langchain