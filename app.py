import os
import streamlit as st
from transformers import pipeline
import warnings
#from dotenv import load_dotenv


st.write(
	"Has environment variables been set:",
	os.environ["HF_ACCESS_TOKEN"] == st.secrets["HF_ACCESS_TOKEN"])

# Title for the web app
st.title("Test Large Language Models from Hugging Face")

# Dropdown for model selection
models = [
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "ibm-granite/granite-3.0-8b-instruct",
    "bartowski/Meta-Llama-3.1-8B-Claude-GGUF"
]

selected_model = st.selectbox("Select a model to test:", models)

# Text input for user prompt
user_input = st.text_area("Enter your text prompt:", "Type something here...")


# Load the selected model using the pipeline
@st.cache_resource
def load_model(model_name):
  with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        access_token = os.getenv("HF_ACCESS_TOKEN")  # Use an environment variable for the access token
        return pipeline("text-generation", model=model_name, framework="tf", use_auth_token=access_token)  # Use TensorFlow framework
        

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
