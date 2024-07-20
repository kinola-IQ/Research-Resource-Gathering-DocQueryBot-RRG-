import os
import streamlit as st 
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv # load environmments an all it's components
import time

load_dotenv()

#load groq and google api key from the .env file
groq_api_key = os.getenv("Groq_api_key")
os.environ['google_api_key'] = os.getenv("google_api_key")

st.title("Q&A chatbot")

#loading the model
llm = ChatGroq(groq_api_key=groq_api_key,model='llama3-8b-8192')

#defining prompt
prompt= ChatPromptTemplate.from_template(
    """
    answer the question based on the context only:
    genrate an accurate response based on the question
    
    <context>
    {context}
    <context>
    question: {input}
    """
    )
payload = st.text_input("enter the documents ou want to embed (pdf only)")


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader(f"{payload}") #Data injestion
        st.session_state.docs = st.session_state.loader.load() #load documents
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) #tokenization
        st.session_state.final = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.final, st.session_state.embeddings) #vectorization
        
def log_history(query,response):
    if "history" not in st.session_state:
        st.session_state.history = []
    st.session_state.history.append({"query":query,"response":response})


prompt1 = st.text_input("what do you wanna know?")

col1, col2 = st.columns([3, 1]) # fro the buttons
with col1:
    if st.button("history"):
        if "history" in st.session_state:
            for entry in st.session_state.history:
                st.write(f"Query: {entry['query']}")
                st.write(f"Response: {entry['response']}")
                st.write("-----------------------------")
        else:
            st.write("No query history available.")    


with col2:  
    if st.button("vector store"):
        vector_embedding()
        st.write("vector_db is ready")


if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    #creating retriever
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain  = create_retrieval_chain(retriever,document_chain)
    
    start = time.process_time()
    response = retrieval_chain.invoke({"input":prompt1})
    log_history(prompt1, response['answer'])
    st.write(response['answer'])
    
    
    
    
    #with streamlit expander
    with st.expander("doc similarity search"):
        #find relevant chunks
        for i,doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("----------------------------")
    
