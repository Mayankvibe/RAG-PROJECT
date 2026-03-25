from langchain_community.document_loaders import PyPDFLoader
data=PyPDFLoader("document_loader/ms.pdf")
docs=data.load()
print(docs)
print(len(docs))