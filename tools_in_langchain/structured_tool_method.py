from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class Multiply(BaseModel):
    a : int = Field(description='first number to multiply')
    b : int = Field(description='Second number to multiply')


def multiply(a, b):
    return a * b


multiply_tool = StructuredTool.from_function(
func = multiply,
description='Multiply two numbers',
name='Multiply',
args_schema=Multiply
)

result = multiply_tool.invoke({'a': 4,'b':6})
print(result)
print(multiply_tool.name)
print(multiply_tool.description)