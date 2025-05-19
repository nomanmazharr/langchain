from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader


directory = DirectoryLoader(
    path="data",
    glob="*.pdf",
    loader_cls=PyPDFLoader
)


loader = directory.lazy_load()

for docs in loader:
    # print(docs.page_content)
    print(docs.metadata)