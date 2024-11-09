import streamlit as st
from transformers import pipeline
import warnings
import torch


# Title for the web app
st.title("Test Large Language Models from Hugging Face")

# Dropdown for model selection
models = [
    "gpt2",
    "facebook/opt-125m",
    "distilgpt2"
]
selected_model = st.selectbox("Select a model to test:", models)

# Text input for user prompt
user_input = st.text_area("Enter your text prompt:", "Type something here...")



# Load the selected model using the pipeline
@st.cache_resource
def load_model(model_name):
    return pipeline("text-generation", model=model_name)

# Button to run the model
if st.button("Generate Response"):
    if user_input:
        try:
            # Load the selected model
            generator = load_model(selected_model)
            
            # Generate text based on the input with truncation enabled
            with st.spinner("Generating response..."):
                result = generator(user_input, max_length=100, num_return_sequences=1, truncation=True)
            
            # Display the result
            st.subheader("Generated Response:")
            st.write(result[0]['generated_text'])
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a text prompt.")
