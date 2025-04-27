from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, load_prompt
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

template = load_prompt("travel_template.json")
# print(template)
chain = template | model
result = chain.invoke({
    "destination_input":"Murree",
    "duration_input":"3 days",
    "interests_input":"nature, hiking",
    "budget_input":"average",
    "style_input":"crazy"
})

print(result.content)