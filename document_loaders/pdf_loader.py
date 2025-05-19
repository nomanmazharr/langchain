from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('practical_mlops.pdf')

mlops_pdf = loader.load()

print(len(mlops_pdf))
print(mlops_pdf[5].page_content)
print(mlops_pdf[60].metadata)