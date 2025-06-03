from langchain_core.tools import tool
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()

@tool
def multiply(a:int, b:int) -> int:
  """Multiply two numbers"""
  return a*b

# print(multiply.invoke({'a':6, 'b':8}))

llm1 = ChatAnthropic(model='claude-3-5-sonnet-latest')

# print(llm1.invoke('hi'))

llm_with_tools = llm1.bind_tools([multiply]) # we can add any number of tools we have in the list

# print(llm_with_tools.invoke('two numbers are 5 and 6'))

query = [HumanMessage("two numbers are 10 and 5")] # saving the Human Message to a list
# print(query)

Aimessage = llm_with_tools.invoke(query)
# print(Aimessage)
# print(Aimessage.tool_calls) # this is what we send to our tools for full result 
# print(Aimessage.tool_calls[0]['args']) # if we only need the content from our tool then we can simply send args value to our tools

query.append(Aimessage) # saving AI message to the list
# print(query)

# tool_result_content = multiply.invoke(Aimessage.tool_calls[0]['args'])
tool_result = multiply.invoke(Aimessage.tool_calls[0])
# print(tool_result_content)
# print(tool_result)

query.append(tool_result) # saving tool_result to the list
# print(query)

# Now we have Human Message, AI Message and Tool Message that we can pass to our model and get the final result

result = llm_with_tools.invoke(query)
print(result)
print(result.content)