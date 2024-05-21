import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun
from langchain import hub
from langchain.agents import create_openai_tools_agent
from langchain.agents import AgentExecutor
from htmlTemplates import css, bot_template, user_template

# Load environment variables
OPENAI_API_KEY = "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"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# Function to initialize wiki and arxiv tools
def wiki_arxiv_extractor(website_input):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

    loader = WebBaseLoader(website_input)
    docs = loader.load()
    documents = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    vectordb = FAISS.from_documents(documents, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
    retriever = vectordb.as_retriever()

    retriever_tool = create_retriever_tool(retriever, "Website", "Search for information in the given website. For any questions about this website, you must use this tool!")

    arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    tools = [wiki, arxiv, retriever_tool]
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_openai_tools_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    return agent_executor

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and update chat history
def handle_userinput(user_question, agent_executor):
    tool_response = agent_executor.invoke({"input": user_question})
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    chat_placeholder = st.empty()  # Create a placeholder for chat history
    chat_html = ""  # Initialize a variable to store the chat history HTML

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            chat_html += user_template.replace("{{MSG}}", message.content)
        else:
            chat_html += bot_template.replace(
                "{{MSG}}", f"Pdf Content: {message.content}<br><br><br>Outsource contents: {tool_response['output']}")
    
    chat_placeholder.write(chat_html, unsafe_allow_html=True)  # Update the chat history

# Function to save chat history as HTML
def save_chat_history(chat_history, filename="chat_history.html"):
    html_content = "<html><head><title>Chat History</title></head><body>"
    for i, message in enumerate(chat_history):
        if i % 2 == 0:
            html_content += user_template.replace("{{MSG}}", message.content)
        else:
            html_content += bot_template.replace("{{MSG}}", message.content)
    html_content += "</body></html>"

    with open(filename, "w") as file:
        file.write(html_content)

    return filename

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    with st.sidebar:
        website_input = st.text_input("Enter Website:")
    st.header("QueryQuill")
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)
                agent_executor = wiki_arxiv_extractor(website_input)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.agent_executor = agent_executor

    if user_question:
        handle_userinput(user_question, st.session_state.agent_executor)

    if st.session_state.chat_history:
        filename = save_chat_history(st.session_state.chat_history)
        with open(filename, "rb") as file:
            st.download_button(
                label="Download chat history",
                data=file,
                file_name=filename,
                mime="text/html"
            )


if __name__ == '__main__':
    main()
