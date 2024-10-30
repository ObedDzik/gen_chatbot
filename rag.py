from langchain_community.document_loaders import DirectoryLoader
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import streamlit as st
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaEmbeddings

loader = DirectoryLoader(r"C:\Users\okdzi\OneDrive\Desktop\feast\Testimonies\Academic_Excellence_Scholarship", glob="**/*.docx")
docs = loader.load()
llm = OllamaLLM(model='llama3.1')
embeddings = OllamaEmbeddings(model="llama3")

RAG_TEMPLATE = """
Use the following pieces of context to answer the question at the end. Retrieve the text directly from the data source. If you don't know the answer, just say that there are no testimonies logged for this category yet. 

<context>
{context}
</context>

Answer the following question:

{question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

vectorstore = InMemoryVectorStore(embeddings)
vectorstore.add_documents(docs)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 1, "fetch_k": 3})

qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

st.title("Retrieve Testimonials Chatbot")
input_txt = st.text_input("Ask your question!")
if input_txt:
    st.write(qa_chain.invoke(input_txt))
# question = "Can you share a testimony on scholarship"

# qa_chain.invoke(question)
