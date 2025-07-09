import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma

# Load secrets from Streamlit
openai_api_key = st.secrets["OPENAI_API_KEY"]
password = st.secrets["PASSWORD"]

pwd = st.sidebar.text_input("Wachtwoord:", value="", type="password")

st.title("📑BBV Chatbot")

# Set a default model
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-4o-mini"

# Load the FAISS vectorstore and set up the retriever and LLM (do this only once)
if "retriever" not in st.session_state or "llm" not in st.session_state:
    vectorstore = Chroma(
        persist_directory="vectorstore_bbv",
        embedding_function=OpenAIEmbeddings()
    )
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model=st.session_state["openai_model"],
        streaming=True
    )
    st.session_state.retriever = retriever
    st.session_state.llm = llm

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if pwd != password:
    st.error("Voer het wachtwoord in om toegang te krijgen tot de chatbot.")
    st.stop()

# React to user input
if prompt := st.chat_input("Stel je vraag over het BBV"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve context and source documents
    docs = st.session_state.retriever.invoke(prompt)
    context = "\n\n".join(doc.page_content for doc in docs)
    system_prompt = (
        "Beantwoord de vraag van de gebruiker op basis van de volgende context uit het BBV:\n"
        f"{context}\n\nVraag: {prompt}\nAntwoord:"
        "Verwijs in je antwoord naar de wettekst van het BBV en de notities."
        "Indien je niet zeker bent of de vraag over het BBV gaat, antwoord: deze vraag gaat niet over het BBV."
    )

    # Stream the answer from the LLM
    with st.chat_message("assistant"):
        response_stream = st.session_state.llm.stream(system_prompt)
        answer = st.write_stream(response_stream)
        # Show sources after the answer
        st.markdown("**Bronnen:**")
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "Onbekend document")
            page = doc.metadata.get("page", "Onbekende pagina")
            st.markdown(f"{i}. {source} p. {page}")

    # Add assistant response to chat history (just the answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})