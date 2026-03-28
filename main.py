from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from  langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

embedding_model=MistralAIEmbeddings(model="mistral-embed")

vectorstore=Chroma(
    persist_directory="chroma_db",
    embedding_function=embedding_model
)

retriver=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

llm=ChatMistralAI(model="mistral-small-2506")

#prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","""you are a helpful ai assitant.
         use only the provided context to answer the question
         if the answer is not present in the context,
         say:"i could not find the answer in the context"
         """),("human","""
Context:{context}
               Question:{question}
""")
    ]
)

print("Rag system created")
print("press 0 to exit ")

while True:
    query=input("You: ")
    if query =="0":
        break

    docs=retriver.invoke(query)

    context="\n\n".join([doc.page_content for doc in docs])

    final_prompt=prompt.invoke({
        "context":context,
        "question":query
    })

    response=llm.invoke(final_prompt)
    print(f"\n AI:{response.content}")