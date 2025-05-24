from langchain_community.retrievers import WikipediaRetriever

retriever = WikipediaRetriever(top_k_results=5)

query = "Sport history between Pakistan and India"

results = retriever.invoke(query)
# print(results)

for i, result in enumerate(results):
    print("Result", {i})
    print("Content \n", {result.page_content})