import streamlit as st
import os
from langchain_community.document_loaders import UnstructuredPDFLoader, UnstructuredCSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
import tempfile

st.set_page_config(page_title="Doc Chat with RAG", layout="centered")
st.title("📄 Ask Questions from Your Document")

# Model settings
model_name = "gemma3:4b"
embedding_model = "nomic-embed-text:v1.5"

uploaded_file = st.file_uploader("Upload a PDF or CSV document", type=["pdf", "csv"])

# Session state to persist processed items
if 'vector_db' not in st.session_state:
    st.session_state.vector_db = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chat_chain' not in st.session_state:
    st.session_state.chat_chain = None

def process_file(file_path, file_type):
    if file_type == "pdf":
        loader = UnstructuredPDFLoader(file_path)
    elif file_type == "csv":
        loader = UnstructuredCSVLoader(file_path)
    else:
        st.error("Unsupported file type.")
        return None

    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = text_splitter.split_documents(data)

    st.success(f"✅ File loaded and split into {len(chunks)} chunks.")

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=OllamaEmbeddings(model=embedding_model),
        collection_name="streamlit-rag",
    )

    return vector_db

def build_chat_chain(vector_db):
    llm = ChatOllama(model=model_name)

    query_prompt = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI language model assistant. Your task is to generate five
        different versions of the given user question to retrieve relevant documents from
        a vector database. Original question: {question}""",
    )

    retriever = MultiQueryRetriever.from_llm(
        vector_db.as_retriever(), llm, prompt=query_prompt
    )

    template = """Answer the question based ONLY on the following context:
    {context}
    Question: {question}"""

    prompt = PromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, chain

if uploaded_file:
    file_type = uploaded_file.type.split("/")[-1]
    
    # Save to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_type}") as tmp:
        tmp.write(uploaded_file.read())
        temp_file_path = tmp.name

    with st.spinner("🔍 Processing your document..."):
        vector_db = process_file(temp_file_path, file_type)

        if vector_db:
            retriever, chain = build_chat_chain(vector_db)
            st.session_state.vector_db = vector_db
            st.session_state.retriever = retriever
            st.session_state.chat_chain = chain

            st.success("✅ Document processed and ready. You can now ask questions below.")
        else:
            st.error("❌ Failed to process the document.")

# Initialize session states
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False
if "answer" not in st.session_state:
    st.session_state.answer = None

question = st.text_input("### Ask a question about the document:")

# When "Get Answer" is clicked, start processing
if not st.session_state.is_processing and st.button("Get Answer", disabled=st.session_state.is_processing):
    if question:
        st.session_state.is_processing = True
        st.rerun()  # Rerun the app to reflect button state

# Show Cancel button if processing
if st.session_state.is_processing:
    cancel = st.button("❌ Cancel")
    if cancel:
        st.session_state.is_processing = False
        st.session_state.answer = "❌ Question cancelled."
        st.rerun()

# Actual processing logic
if st.session_state.is_processing and not st.session_state.answer:
    with st.spinner("🧠 Thinking..."):
        try:
            answer = st.session_state.chat_chain.invoke(input=question)
            st.session_state.answer = answer
            st.session_state.is_processing = False
            st.rerun()
        except Exception as e:
            st.session_state.answer = f"Error: {e}"
            st.session_state.is_processing = False
            st.rerun()

# Show the answer
if st.session_state.answer:
    st.markdown(f"### 📝 Answer:\n{st.session_state.answer}")
