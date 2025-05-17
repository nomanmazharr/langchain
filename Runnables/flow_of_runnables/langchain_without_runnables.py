import random

class LangchainLLM():
    def __init__(self):
        print("LLM Initialized")

    def predict(self, prompt):
        responses = [
            "Pakistan is a country in South Asia.",
            "Chinese is a language spoken in China.",
            "The capital of France is Paris.",
            "The sun is a star at the center of the solar system.",
            "The moon orbits the Earth."
        ]

        return {"output": random.choice(responses)}


class PromptTemplate():
    def __init__(self, template, input_variables):
        self.template = template
        self.input_variables = input_variables

    def format(self, input_dict):
        return self.template.format(**input_dict)



llm = LangchainLLM()
# result = llm.predict("what is the capital of France?")
# print(result)

prompt1 = PromptTemplate(
    template="tell me about {country} which is located in {location}",
    input_variables=["country", "location"]
)


final_prompt = prompt1.format({"country": "Pakistan", "location": "Asia"})

print(final_prompt)

result = llm.predict(final_prompt)
print(result)