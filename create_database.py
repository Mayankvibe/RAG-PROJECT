# load pdf
# split into chunks 
# create the embeddings
# store into chroma 
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv()

data=PyPDFLoader("document_loader/ML_book.pdf")
docs=data.load()

splitter=RecursiveCharacterTextSplitter(
   chunk_size=1000,
    chunk_overlap=100 

)
chunks=splitter.split_documents(docs)

embedding_model=MistralAIEmbeddings(model="mistral-embed")
vectorstore=Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)