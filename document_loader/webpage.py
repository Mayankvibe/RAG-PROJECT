from langchain_community.document_loaders import WebBaseLoader
data=WebBaseLoader("https://docs.mistral.ai/")
docs=data.load()
print(docs)
print(len(docs))