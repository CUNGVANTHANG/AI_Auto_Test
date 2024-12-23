import os
import pandas as pd
import streamlit as st
import json
from model_llm import ChatModelLLM
from model_gemini import ChatModelGemini
from context import DEFAULT_CONTEXT
import rag_llm

FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)

@st.cache_resource
def load_model(answer_model_option):
    model = None
    if answer_model_option == "gemma-2b-it":
        model = ChatModelLLM(model_id="google/gemma-2b-it", device="cpu")
    else:
        model = ChatModelGemini(model_id="gemini-1.5-pro", device="cpu")
    return model

@st.cache_resource
def load_encoder(embedding_model_option):
    encoder = None
    if embedding_model_option == "all-MiniLM-L12-v2":
        encoder = rag_llm.Encoder(
            model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu"
        )
    else:    
        encoder = None
    return encoder

def process_files(uploaded_files, encoder):
    file_paths = []
    DB = None
    if uploaded_files:
        with st.spinner("Processing the files..."):
            for uploaded_file in uploaded_files:
                file_paths.append(save_file(uploaded_file))
            if file_paths:  # Chỉ tiếp tục nếu có file hợp lệ
                docs = rag_llm.load_and_split_files(file_paths)
                if docs:  # Kiểm tra danh sách tài liệu
                    print(docs)
                    DB = rag_llm.FaissDb(docs=docs, embedding_function=encoder.embedding_function)
                else:
                    st.error("No documents were processed. Please check your files.")
            else:
                st.error("No files were uploaded.")
        st.success("Files processed successfully!")
    return DB 

def save_file(uploaded_file):
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def handle_user_input(prompt, model, DB, k, max_new_tokens):
    if not prompt.strip(): 
        combined_prompt = DEFAULT_CONTEXT
    else:  
        combined_prompt = f"{prompt}\n\n{DEFAULT_CONTEXT}"
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt if prompt.strip() else "Using default context.")

    df = None
    with st.chat_message("assistant"):
        context = None if not DB else DB.similarity_search(combined_prompt, k=k)
        answer = model.generate(combined_prompt, context=context, max_new_tokens=max_new_tokens)
        print(answer)
        try:
            cleaned_json = answer.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned_json)
            df = pd.DataFrame(data)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error processing JSON data: {e}")
        # st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": df})

def setup_page():
    st.set_page_config( 
        page_title="AI Auto Test",  
        page_icon="💬",  
        layout="centered"
    )

def display_title():
    st.title("💬 AI Auto Test")
    st.caption("🚀 AI Automatic Test Case Generation")

def display_sidebar():
    with st.sidebar:
        st.header("Settings")

         # Add option to select input mode
        input_mode = st.radio(
            "Input Mode",
            options=["Chat Input", "Button Prompt"],
            index=1,
            help="Choose how to provide input to the assistant."
        )

        # # Radio button to choose embedding model
        # embedding_model_option = st.radio(
        #     "🔤 Choose an Embedding Model",
        #     options=["all-MiniLM-L12-v2", "gemini-1.5-pro"],
        #     index=0, # Default to first model
        # )

        embedding_model_option = "all-MiniLM-L12-v2"

        encoder = load_encoder(embedding_model_option)

        # # Radio button to choose answer generation model
        # answer_model_option = st.radio(
        #     "🤖 Choose an Answer Generation Model",
        #     options=["gemma-2b-it", "gemini-1.5-pro"],
        #     index=1, # Default to first model
        # )

        answer_model_option = "gemini-1.5-pro"

        model = load_model(answer_model_option)

        # max_new_tokens = st.number_input("max_new_tokens", 128, 4096, 512)
        max_new_tokens = 512
        # k = st.number_input("k", 1, 10, 3)
        k = 3

        uploaded_files = st.file_uploader(
            "Upload PDFs or Word documents for context", type=["pdf", "doc", "docx"], accept_multiple_files=True
        )

        DB = process_files(uploaded_files, encoder)

        return model, max_new_tokens, k, DB, input_mode

def initialize_chat_history():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def display_chat_messages():
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                st.dataframe(message["content"])
            else:
                st.markdown(message["content"])

def main():
    setup_page()
    display_title()

    model, max_new_tokens, k, DB, input_mode = display_sidebar()

    initialize_chat_history()

    display_chat_messages()

    if input_mode == "Chat Input":
        if prompt := st.chat_input("Ask me anything!"):
            handle_user_input(prompt, model, DB, k, max_new_tokens)
    else:
        if st.button("Generate Test Case Requirements"):
            handle_user_input("write test case requirement", model, DB, k, max_new_tokens)

if __name__ == "__main__":
    main()