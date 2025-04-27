from langchain.prompts import ChatPromptTemplate


template = ChatPromptTemplate([
    ("system", "You are a helpful assistant that tells user about {sport}"),
    ("user", "who is {person}")
])

prompt = template.invoke({"person":"Shahid Afridi", "sport":"cricket"})


print(prompt)