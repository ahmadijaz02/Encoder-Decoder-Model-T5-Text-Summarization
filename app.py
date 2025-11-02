import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import os
import torch

MODEL_DIR = "t5-summarizer-with-checkpoints" 

@st.cache_resource
def load_model_and_tokenizer(model_directory):
    """
    Loads the T5 model and tokenizer from a local directory.
    Uses st.cache_resource to load only once.
    """
    st.write(f"Loading model from {model_directory}...")
    try:
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
    # T5 models require a prefix, like "summarize: "
    prefixed_text = "summarize: " + text.strip()

    # Encode the text
    inputs = tokenizer.encode(
        prefixed_text, 
        return_tensors="pt", 
        max_length=512, 
        truncation=True
    )

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
st.title("📄 Text Summarization App")
st.write("This app uses a fine-tuned T5 model to summarize your text.")
st.write(f"Model directory: `{MODEL_DIR}`")

# Check if model directory exists
if not os.path.exists(MODEL_DIR):
    st.error(f"Model directory not found: '{MODEL_DIR}'")
    st.error("Please make sure your downloaded model folder is in the same directory as this script and is named correctly.")
else:
    # Load the model
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
                with st.spinner("Summarizing..."):
                    try:
                        summary = generate_summary(input_text, model, tokenizer)
                        
                        st.subheader("Generated Summary:")
                        st.success(summary)
                    except Exception as e:
                        st.error(f"An error occurred during summarization: {e}")
