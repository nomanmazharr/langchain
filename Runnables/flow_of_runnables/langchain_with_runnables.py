from abc import ABC, abstractmethod
import random


class Runnable(ABC):

    @abstractmethod
    def invoke(self, input_data):
        pass


class LangchainLLM(Runnable):
    def __init__(self):
        print("LLM Initialized")

    def invoke(self, prompt):
        responses = [
            "Pakistan is a country in South Asia.",
            "Chinese is a language spoken in China.",
            "The capital of France is Paris.",
            "The sun is a star at the center of the solar system.",
            "The moon orbits the Earth."
        ]

        return {"output": random.choice(responses)}

    def predict(self, prompt):
        responses = [
            "Pakistan is a country in South Asia.",
            "Chinese is a language spoken in China.",
            "The capital of France is Paris.",
            "The sun is a star at the center of the solar system.",
            "The moon orbits the Earth."
        ]

        return {"output": random.choice(responses)}


class PromptTemplate(Runnable):
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_dict):
        return self.template.format(**input_dict)

    def format(self, input_dict):
        return self.template.format(**input_dict)


class OutputParser(Runnable):
    def __init__(self):
        pass

    def invoke(self, data):
        return data["output"]


class RunnableConnector(Runnable):
    def __init__(self, runnables):
        self.runnables = runnables


    def invoke(self, input_data):
        for runnable in self.runnables:
            input_data = runnable.invoke(input_data)

        return input_data


llm = LangchainLLM()
# result = llm.predict("what is the capital of France?")
# print(result)

prompt1 = PromptTemplate(
    template="tell me about {country} which is located in {location}",
    input_variables=["country", "location"]
)

parser1 = OutputParser()

# final_prompt = prompt1.format({"country": "Pakistan", "location": "Asia"})

# print(final_prompt)

# result = llm.predict(final_prompt)
# print(result)

chain = RunnableConnector([prompt1, llm])
chain = RunnableConnector([prompt1, llm, parser1])

result = chain.invoke({"country": "Pakistan", "location": "Asia"})

print(result)