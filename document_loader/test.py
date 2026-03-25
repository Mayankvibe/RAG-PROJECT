from langchain_community.document_loaders import TextLoader
data=TextLoader("document_loader/note.txt")
result=data.load()
print(result[0].metadata)
print(len(result))