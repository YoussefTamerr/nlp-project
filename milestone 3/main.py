import faiss
from dotenv import load_dotenv
import os
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from fastapi import FastAPI
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from datasets import load_from_disk


# Load previously saved dataset
samples = load_from_disk("narrativeqa_1000")

ds_train_context = [row['document']['summary']['text'] for row in samples]
ds_train_question = [row['question']['text'] for row in samples]
ds_train_answer = [row['answers'][0]['text'] for row in samples]

print("Loaded dataset with {} samples.".format(len(samples)))

load_dotenv()
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

print("Setup FAISS vectorstore with embedding model.")

documents = [Document(page_content=doc, metadata={"source": "test"}) for doc in ds_train_context]
# documents = [
#     Document(page_content="kimo is an asian", metadata={"source": "test"}),
# ]

texts = text_splitter.split_documents(documents)

vectorstore.add_documents(texts)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

print("added documents to FAISS vectorstore.")

memory_llama = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=4
)

memory_gemma = ConversationBufferWindowMemory(
    memory_key="chat_history",
    return_messages=True,
    k=4
)

system_message = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant."
    "You are given a context and a question."
    "You should answer the question based on the context."
    "If the context does not contain the answer, say 'I don't know'."
    "Dont answer the question with a question."
    "Be specific and concise."
)

human_message = HumanMessagePromptTemplate.from_template(
    "Context:\n{context}\n\nQuestion:\n{question}"
)

chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

llm_llama = ChatOpenAI(
    model="llama3-8b-8192",
    temperature=0,
    max_tokens=1000,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_URL"),
)

qa_llama = ConversationalRetrievalChain.from_llm(
    llm=llm_llama,
    chain_type="stuff",
    retriever=retriever,
    memory=memory_llama,
    combine_docs_chain_kwargs={"prompt": chat_prompt},
)

llm_gemma = ChatOpenAI(
    model="gemma2-9b-it",
    temperature=0,
    max_tokens=1000,
    api_key=os.getenv("API_KEY"),
    base_url=os.getenv("API_URL"),
)

qa_gemma = ConversationalRetrievalChain.from_llm(
    llm=llm_gemma,
    chain_type="stuff",
    retriever=retriever,
    memory=memory_gemma,
    combine_docs_chain_kwargs={"prompt": chat_prompt},
)

print("Setup ConversationalRetrievalChain with llama model.")
print("Setup ConversationalRetrievalChain with gemma model.")

last_model = {"name": 'gemma'}


app = FastAPI()
@app.get("/chat")
def chat(prompt: str, model: str):
    if model == "llama":
        qa = qa_llama
    elif model == "gemma":
        qa = qa_gemma
    else:
        return {"error": "Invalid model name. Use 'llama' or 'gemma'."}
    
    if last_model["name"] != model and last_model["name"] is not None:
        last_model["name"] = model
        if model == "llama":
            qa.memory.clear()
        elif model == "gemma":
            qa.memory.clear()

    response = qa.invoke(prompt)
    print("Response:", response)
    return {"response": response['answer']}



# def main():
#     # Example query
#     query = "is kimo asian?"
#     result = qa.invoke(query)
#     print("Result:", result['chat_history'])


#     # query2 = "what race is he?"
#     # result2 = qa.invoke(query2)
#     # print("Result:", result2['chat_history'])


# if __name__ == "__main__":
#     main()
