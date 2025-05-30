import os
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain_voyageai import VoyageAIEmbeddings

# set your apis
load_dotenv()
anthropic_api = os.getenv('ANTHROPIC_API_KEY')
voyageai_api = os.getenv('VOYAGE_API_KEY')
google_api = os.getenv('GEMINI_API_KEY')

# Loading transcript from youtube video
id = "JMYQmGfTltY"
try:
  transcripts_list = YouTubeTranscriptApi.get_transcript(video_id=id, languages=['en'])
  transcript = " ".join(chunk['text'] for chunk in transcripts_list)
#   print(transcript)
except TranscriptsDisabled:
  print('No transcript available for this video')

# splitting data usign text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.create_documents([transcript])
# print(len(chunks))

# for chunk in chunks:
#   print(chunk)
# print(chunks[10].page_content)
# print(chunks[11].page_content)
# print(chunks[12].page_content)

# Creating embeddings using openai embedding model
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
# embeddings = VoyageAIEmbeddings(api_key=voyageai_api, model='voyage-code-3')
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001', google_api_key=google_api)
vector_stores = FAISS.from_documents(chunks, embeddings)
# checking vector store by indexing vectors by their ids
# print(vector_stores.index_to_docstore_id)
# vector_stores.get_by_ids(['2175371a-2180-4635-837d-c90b0e97c2d3'])


# # Using retriever to retrieve the best matching documents on the basis of similarity
retriever = vector_stores.as_retriever(search_type='similarity', search_kwargs={'k':4})
# retriever
retriever.invoke('What is AI')

# # model initialization
# model = ChatOpenAI(model='gpt-4o-mini', temperature=0.2)
model2 = ChatAnthropic(model='claude-3-5-sonnet-latest', api_key=anthropic_api)

# # defining prompt
prompt = PromptTemplate(
    template='''You are an AI assistant that only answers based on the content provided in the transcript. If the query cannot be answered using the given transcript, respond with:
    "I have no idea."
    Do not use any outside knowledge or assumptions. Only use the information explicitly stated in the transcript.
    Transcript: {transcript}
    Query: {query}''',
    input_variables= ['transcript', 'query']
)

# # preparing query and retrieving data from retriever
query ='tell me about Salman Khan'
retrieved_docs = retriever.invoke(query)
# retrieved_docs
retrieved_results = "\n".join(doc.page_content for doc in retrieved_docs)
# retrieved_results

# preparing prompt with input variables
final_prompt = prompt.invoke({'transcript': retrieved_results, 'query': query})
# print(final_prompt)

# # model invoking
# results = model.invoke(final_prompt)
results = model2.invoke(final_prompt)
# print(results.content)


# # Chain to automate the whole process

# defining a parser
parser = StrOutputParser()

# # defining a function for document retrieval
def format_docs(documents):
  retrieved_docs = "\n".join(doc.page_content for doc in documents)
  return retrieved_docs

# # creating parallel chain to get the transcript and query for prompt
parallel_chain = RunnableParallel({
    'transcript': retriever | RunnableLambda(format_docs),
    'query' : RunnablePassthrough()
})

# # invoking chain ang getting transcript and query to pass to the prompt
parallel_chain.invoke('are we living in aliens')

# # main chain initializer
chain = parallel_chain | prompt | model2 | parser
# invoking the final chain
result = chain.invoke('are aliens living among us?')
print(result)