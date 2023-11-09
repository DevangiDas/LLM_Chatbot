# File Name: app.py
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import time
from sklearn.metrics import mean_squared_error
from sklearn.metrics import pairwise
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from htmlTemplates import css, bot_template, user_template

# Define reference responses for evaluation
reference_responses = ["Reference response 1",
                       "Reference response 2", "Reference response 3"]


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def evaluate_vectorization_quality(vectorstore):
    vectors = vectorstore.get_vectors()
    similarity_matrix = pairwise.cosine_similarity(vectors)
    mse = mean_squared_error(similarity_matrix, similarity_matrix)
    return mse


def evaluate_response(reference, generated):
    bleu = sentence_bleu([reference.split()], generated.split())
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    rouge_l_score = scores['rougeL'].fmeasure
    return bleu, rouge_l_score


def handle_user_input(user_question, reference_responses, chat_history):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            user_response = message.content
            st.write(user_template.replace(
                "{{MSG}}", user_response), unsafe_allow_html=True)

            if i + 1 < len(st.session_state.chat_history):
                bot_response = st.session_state.chat_history[i + 1].content
                st.write(bot_template.replace(
                    "{{MSG}}", bot_response), unsafe_allow_html=True)

                bleu_score, rouge_score = evaluate_response(
                    reference_responses[i // 2], bot_response)
                st.write(f"BLEU Score: {bleu_score:.2f}")
                st.write(f"ROUGE-L Score: {rouge_score:.2f}")


def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    load_dotenv()
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    user_question = st.text_input(
        "Ask a question about your documents:", key="user_question")

    reference_responses = ["Reference response 1",
                           "Reference response 2", "Reference response 3"]

    if user_question:
        handle_user_input(user_question, reference_responses,
                         st.session_state.chat_history)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                start_time = time.time()
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                vectorization_quality = evaluate_vectorization_quality(
                    vectorstore)
                st.write(
                    f"Vectorization Quality (MSE): {vectorization_quality:.4f}")
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)
                end_time = time.time()
                execution_time = end_time - start_time
                st.write(f"Execution Time: {execution_time:.2f} seconds")


if __name__ == '__main__':
    main()
