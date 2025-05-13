import faiss
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from fastapi import FastAPI

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

index = faiss.IndexFlatL2(len(embedding_model.embed_query("tikha")))
docstore = InMemoryDocstore()

vectorstore = FAISS(
    index=index,
    embedding_function=embedding_model,
    docstore=docstore,
    index_to_docstore_id={},
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len,
)

documents = [
    Document(page_content="kimo's race is subafrican but sometimes he is mistaken for being indian, kimo would like to be asian so that the asian baddies would want him", metadata={"source": "test"})
]

texts = text_splitter.split_documents(documents)

vectorstore.add_documents(texts)

load_dotenv()
llm = ChatOpenAI(
    # model="allam-2-7b",
    model="deepseek-r1-distill-llama-70b",
    temperature=0,
    max_tokens=1000,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_URL"),
    stream=True,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

memory = ConversationBufferMemory(memory_key="chat_history")

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    memory=memory,
)

SYSTEM_PROMPT = (
    "You are a helpful assistant."
)

app = FastAPI()
@app.get("/chat")
def chat(prompt: str, model: str):
    if model == "LLM-1":
        chosen_qa = qa
    elif model == "LLM-2":
        chosen_qa = qa
    else:
        return {"error": "Invalid model selected."}
    
    if not memory.load_memory_variables({}).get("chat_history"):
        memory.chat_memory.add_ai_message(SYSTEM_PROMPT)
    
    response = qa.invoke(prompt)
    return {"response": response}



# def main():
#     # Example query
#     query = "who is kimo?"
#     result = qa.invoke(query)
#     query2 = "what race is he?"
#     result2 = qa.invoke(query2)
#     print("Query:", query)
#     print("Result:", result)
#     print("Query:", query2)
#     print("Result:", result2)


# if __name__ == "__main__":
#     main()
