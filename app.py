import streamlit as st
import os
from llama_index import VectorStoreIndex, ServiceContext, Document
from llama_index.llms import OpenAI
import openai
from llama_index import SimpleDirectoryReader

st.set_page_config(page_title="Chat with Hawker Test Bot 🤖💬 powered by LlamaIndex 🦙", layout="centered", initial_sidebar_state="auto", menu_items=None)
openai.api_key = os.getenv("OPENAI_API_KEY")
st.title("Chat with Hawker Test Bot 🤖💬")

with st.sidebar:
    st.title("Powered by LlamaIndex 🦙")
    st.markdown('''
  
                ''')
    # Create a file uploader for files
    uploaded_file = st.sidebar.file_uploader('Select a file for upload to knowledge base')

    # Add a submit button in the sidebar
    if uploaded_file is not None:
        # Get the path of the selected subdirectory
        save_path = os.path.join(f'./data')
        # Save the uploaded file in the selected subdirectory
        with open(os.path.join(save_path, uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            # Run the custom function
            st.success('File uploaded successfully.')
         
if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, Human! Ask me a question related to Hawker!"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Hawker related docs – hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo", temperature=0.5, system_prompt="You are an expert on hawker centres in Singapore and your job is to answer questions about hawker centres. Assume that all questions are related to hawker centres. Keep your answers based on facts – do not hallucinate."))
        index = VectorStoreIndex.from_documents(docs, service_context=service_context)
        return index

index = load_data()
# chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True, system_prompt="You are an expert on hawker centres in Singapore and your job is to answer questions about hawker centres. Assume that all questions are related to hawker centres. Keep your answers based on facts – do not hallucinate.")

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)

if prompt := st.chat_input("Your question"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
