import os
from langchain_community.tools import DuckDuckGoSearchResults, Tool
from langchain_community.utilities import SerpAPIWrapper
from dotenv import load_dotenv

load_dotenv()

serper_api = os.getenv("SERPER_API_KEY")
# print(f"Loaded API Key: {serper_api}")
search_tool1 = DuckDuckGoSearchResults()
# loading as utility
search_utility = SerpAPIWrapper(serpapi_api_key=serper_api)
# results = search_tool1.invoke('Tell me about latest law student case in india who got arrested')
# results = search_tool1.invoke('remarks on latest india Pakistan war who won?')
# results = search_utility.run('latest advancements in Artificial intelligence')

# configuring serpapi as a tool too 
search_tool_serp = Tool(
    name='serp_api',
    func=search_utility.run,
    description= 'A google search utility tool that use serp api to search the latest info on google'
)

results = search_tool_serp.invoke('latest advancements in Artificial intelligence')
print(results)
# print(search_tool1.name)
# print(search_tool1.description)
# print(search_tool1.args)