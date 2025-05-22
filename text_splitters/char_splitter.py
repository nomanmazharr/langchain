from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('mlops.pdf')

docs = loader.load()

# print(len(docs))

splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap = 0,
    separator=''
)

result = splitter.split_documents(docs)

# result = splitter.split_text("""
# We’ve reached a turning point in the story of machine learning where the technology
#  has moved from the realm of theory and academics and into the “real world”—that is,
#  businesses providing all kinds of services and products to people across the globe.
#  While this shift is exciting, it’s also challenging, as it combines the complexities of
#  machine learning models with the complexities of the modern organization.
#  One difficulty, as organizations move from experimenting with machine learning to
#  scaling it in production environments, is maintenance. How can companies go from
#  managing just one model to managing tens, hundreds, or even thousands? This is not
#  only where MLOps comes into play, but it’s also where the aforementioned complexi
# ties, both on the technical and business sides, appear. This book will introduce read
# ers to the challenges at hand, while also offering practical insights and solutions for
#  developing MLOps capabilities.
#  Who This Book Is For
#  We wrote this book specifically for analytics and IT operations team managers, that
#  is, the people directly facing the task of scaling machine learning (ML) in production.
#  Given that MLOps is a new field, we developed this book as a guide for creating a
#  successful MLOps environment, from the organizational to the technical challenges
#  involved.
#  How This Book Is Organized
#  This book is divided into three parts. The first is an introduction to the topic of
#  MLOps, diving into how (and why) it has developed as a discipline, who needs to be
#  involved to execute MLOps successfully, and what components are required.
#  The second part roughly follows the machine learning model life cycle, with chapters
#  on developing models, preparing for production, deploying to production, monitor
# ing, and governance. These chapters cover not only general considerations, but
# """)

print(result[27])
# print(result)