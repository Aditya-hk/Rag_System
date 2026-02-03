import streamlit as st
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader

st.title("PDF RAG Q&A")

# ✅ Correct API Key way
os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]

# ✅ Streamlit Upload (NOT Colab)
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:

    # Save uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    text = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=50
    )

    docs = splitter.split_documents(text)

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(docs, embedding)

    retriver = vectorstore.as_retriever(search_kwargs={"k":3})

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    prompt = PromptTemplate(
        input_variables=["context","question"],
        template = """
        USE ONLY CONTEXT BELOW TO ANSWER THE QUESTIONS

        CONTEXT:
        {context}

        QUESTION: {question}

        ANSWER:
        """
    )

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    def rag_pipeline(query):
        r_docs = retriver.invoke(query)
        context = format_docs(r_docs)
        full_prompt = prompt.format(context=context, question=query)
        response = llm.invoke(full_prompt)
        return response.content

    query = st.text_input("Ask Question")

    if query:
        answer = rag_pipeline(query)
        st.write(answer)
