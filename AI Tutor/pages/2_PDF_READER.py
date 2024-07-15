import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from transformers import pipeline
import os
import threading
import time
import sys
from melo.api import TTS
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from htmlTemplates import css, bot_template, user_template



# Function to stop the TTS playback
class StopSignal:
    def __init__(self):
        self.stop = False

stop_signal = StopSignal()

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOllama(model="llama3", temperature="0.5")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Load model directly
from transformers import AutoModel
tts_pipeline = AutoModel.from_pretrained("myshell-ai/MeloTTS-English")

def read_text_aloud(text, stop_signal):
    chunks = text.split('\n')
    for chunk in chunks:
        if stop_signal.stop:
            break
        audio = tts_pipeline(chunk)
        audio_file = "output.wav"
        with open(audio_file, "wb") as f:
            f.write(audio["wav"])
        audio_bytes = open(audio_file, "rb").read()
        st.audio(audio_bytes, format="audio/wav")
        time.sleep(0.1)

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Your Tutor is Here", page_icon=":books:")

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "current_word" not in st.session_state:
        st.session_state.current_word = 0

    st.header("Tutor :books:")
    user_question = st.text_input("Ask question: ")
    if user_question:
        handle_userinput(user_question)

    st.subheader("Your documents: ")
    pdf_docs = st.file_uploader("Upload your PDFs", accept_multiple_files=True)
    if st.button("Process"):
        with st.spinner("Processing..."):
            # Get raw text
            raw_text = get_pdf_text(pdf_docs)

            # Display the extracted text
            st.subheader("Extracted Text:")
            st.write(raw_text)

            # Read the text aloud
            stop_signal.stop = False
            threading.Thread(target=read_text_aloud, args=(raw_text, stop_signal)).start()

            # Get text chunks
            text_chunks = get_text_chunks(raw_text)

            # Create vector store
            vectorstore = get_vectorstore(text_chunks)

            # Create conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)

    if st.button("Stop"):
        stop_signal.stop = True
        

if __name__ == '__main__':
    main()
