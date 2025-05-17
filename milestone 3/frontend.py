import streamlit as st
import requests



MODEL_OPTIONS = ["Gemma 2 9B-IT", "LLaMA 3 8B-8192"]

if "selected_model" not in st.session_state:
    st.session_state.selected_model = MODEL_OPTIONS[0]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

selected_model = st.selectbox("Choose a model", MODEL_OPTIONS)

if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    st.session_state.chat_history = [] 
    st.rerun()


st.title("Chatbot")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask me anything about blabla")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    bot_response = ""

    # Call the API
    model_name_to_send = "gemma"
    if st.session_state.selected_model == "Gemma 2 9B-IT": 
          model_name_to_send = "gemma" 
    else: 
          model_name_to_send = "llama"
    response = requests.get(
        "http://localhost:8000/chat",
        params={"prompt": user_input, "model": model_name_to_send},
    )

    if response.status_code == 200:
        data = response.json()
        if "response" in data:
            bot_response = data["response"]
        else:
            bot_response = "Error: No response from the API."

        with st.chat_message("assistant"):
            st.markdown(bot_response)

        st.session_state.chat_history.append({"role": "assistant", "content": bot_response})
        
    else:
        st.error("Error: Unable to get a response from the API.")