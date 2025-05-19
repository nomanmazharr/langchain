from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
model = ChatOpenAI()


loader = WebBaseLoader("https://www.amazon.com/Garmin-f%C4%93nix%C2%AE-Multisport-Long-Lasting-Built/dp/B0DD622CGH?ref=dlx_memor_dg_dcl_B0DD622CGH_dt_sl14_ef_pi&pf_rd_r=633WRRJVN4PN31V6GK1F&pf_rd_p=e5de50e6-9ed5-43e0-b855-5a9e59492def&th=1")

site_content = loader.load()

prompt = PromptTemplate(
    template="what product are we talking about and what's it's price in USD and PKR? \n {text}",
    input_variables=['text']
)
parser = StrOutputParser()

# print(len(site_content))
# print(site_content[0].metadata)

chain = prompt | model | parser
result = chain.invoke({"text": site_content[0].page_content})
print(result)