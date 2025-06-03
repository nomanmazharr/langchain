from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Type


class Multiply(BaseModel):
  a : int = Field(description='first number to multiply')
  b : int = Field(description='Second number to multiply')


class MultiplyNumbers(BaseTool):
  name: str = 'multiply'
  description: str = "Multiply two numbers"

  args_schema: Type[BaseModel] = Multiply

  def _run(self, a, b):
    return a*b

multiply_basetool = MultiplyNumbers()
results = multiply_basetool.invoke({'a': 8, 'b': 10})
print(results)


print(multiply_basetool.name)
print(multiply_basetool.description)

print(multiply_basetool.args)