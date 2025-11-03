import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import os
import zipfile
import sys

# --- CONFIGURATION ---
# Kaggle dataset details (from your link)
KAGGLE_DATASET = "ahmadijaz92/genai-p3-t2"
# The name of the folder *inside* the Kaggle dataset
MODEL_DIR = "t5-summarizer-with-checkpoints" 

def download_model_from_kaggle():
    """
    Checks if the model is downloaded. If not, downloads it from Kaggle
    using Streamlit secrets.
    """
    # 1. Check if model already exists
    if os.path.exists(MODEL_DIR):
        st.write("Model directory already exists. Skipping download.")
        return

    # 2. Check for Streamlit secrets
    if 'KAGGLE_USERNAME' not in st.secrets or 'KAGGLE_KEY' not in st.secrets:
        st.error("Kaggle credentials not found in Streamlit secrets.")
        st.error("Please add KAGGLE_USERNAME and KAGGLE_KEY to your app's secrets.")
        st.stop()

    st.info(f"Model not found locally. Downloading from Kaggle dataset: {KAGGLE_DATASET}...")

    # 3. Set up Kaggle API credentials
    os.environ['KAGGLE_USERNAME'] = st.secrets['KAGGLE_USERNAME']
    os.environ['KAGGLE_KEY'] = st.secrets['KAGGLE_KEY']

    # 4. Import Kaggle API and download
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()

        # Download and unzip the dataset files into the current directory
        api.dataset_download_files(KAGGLE_DATASET, path='.', unzip=True)
        
        st.success(f"Model downloaded and unzipped to '{MODEL_DIR}'")
    
    except Exception as e:
        st.error(f"Error downloading from Kaggle: {e}")
        st.stop()

@st.cache_resource
def load_model_and_tokenizer(model_directory):
    """
    Loads the T5 model and tokenizer from a local directory.
    Uses st.cache_resource to load only once.
    """
    st.write(f"Loading model from local directory: {model_directory}...")
    try:
        # T5Tokenizer requires the 'sentencepiece' library
        tokenizer = T5Tokenizer.from_pretrained(model_directory)
        model = T5ForConditionalGeneration.from_pretrained(model_directory)
        st.write("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error(f"Please make sure the directory '{model_directory}' exists and contains the model files.")
        return None, None

def generate_summary(text, model, tokenizer):
    """
    Generates a summary for the given text using the loaded model.
    """
    # T5 models require a prefix
    prefixed_text = "summarize: " + text.strip()

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Encode the text
    inputs = tokenizer.encode(
        prefixed_text,  
        return_tensors="pt",  
        max_length=512,  # Use the same max_length as training
        truncation=True
    ).to(device)

    # Generate the summary
    summary_ids = model.generate(
        inputs,  
        max_length=150,  # Max length of the summary
        min_length=40,   # Min length of the summary
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )

    # Decode the summary
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# --- Streamlit App UI ---

st.set_page_config(page_title="Text Summarizer", layout="wide")
st.title("📄 T5 Text Summarization App")
st.write(f"This app uses a fine-tuned T5 model from Kaggle (`{KAGGLE_DATASET}`).")

# 1. Download the model (runs only if model isn't downloaded)
download_model_from_kaggle()

# 2. Load the model from the local directory
model, tokenizer = load_model_and_tokenizer(MODEL_DIR)

if model and tokenizer:
    st.success("Model and tokenizer are loaded and ready!")

    # User input text area
    input_text = st.text_area(
        "Enter text to summarize:",  
        height=250,  
        placeholder="Paste a long article or text here..."
    )

    # Summarize button
    if st.button("Generate Summary"):
        if not input_text.strip():
            st.warning("Please enter some text to summarize.")
        else:
            # Show a spinner while processing
            with st.spinner("Summarizing... (This may take a moment)"):
                try:
                    summary = generate_summary(input_text, model, tokenizer)
                    st.subheader("Generated Summary:")
                    st.success(summary)
                except Exception as e:
                    st.error(f"An error occurred during summarization: {e}")

else:
    st.error("Failed to load the model. The app cannot continue.")
    st.stop()

