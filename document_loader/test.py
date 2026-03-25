from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
splitter=CharacterTextSplitter(
    separator="",
    chunk_size=10,
    chunk_overlap=1
)
data=TextLoader("document_loader/note.txt")
result=data.load()
chunks=splitter.split_documents(result)
print(chunks)
print(len(chunks))