# There are multiple methods in langchian to create own tool

from langchain_community.tools import tool

# Steps to create tool using Tool from langchain

# Step -1 .... Creating a function

def multiply(a, b):
    return a*b

# Step -2 .... Assigning type to the variables
def multiply(a:int, b: int) -> int:
    return a*b

# Step -3 .... Intialize function with decorator
@tool
def multiply(a:int, b: int) -> int:
    """A function to multiply two numbers"""
    return a*b


# In the same way we invoke other tools we can invoke custom build tools too: 
result = multiply.invoke({'a': 4, 'b':5})
print(result)
# print(multiply.name)
# print(multiply.args)
# print(multiply.description)
print(multiply.args_schema.model_json_schema())