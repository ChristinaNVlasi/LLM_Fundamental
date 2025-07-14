import streamlit as st

#UI Initialization
st.title("My Local GPT 🤭")

# Managing Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Displaying Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#Receiving User Input
if prompt := st.chat_input("Write something"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Displaying User Message
with st.chat_message("user"):
    st.markdown(prompt)

from langchain_ollama import ChatOllama

with st.chat_message("assistant"):
        local_model = "llama3.2"
        llm = ChatOllama(model=local_model, temperature=0.7)

stream = llm.stream(
    input=[
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]
)
response = st.write_stream(stream)

st.session_state.messages.append({"role": "assistant", "content": response})