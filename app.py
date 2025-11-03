import streamlit as st
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch

# --- CONFIGURATION ---
# IMPORTANT: Update this to your Hugging Face model name
MODEL_NAME = "ahmadijaz92/genai-p3-t2-summarizer" 

@st.cache_resource
def load_model_and_tokenizer(model_name):
    """
    Loads the T5 model and tokenizer from Hugging Face Hub.
    Uses st.cache_resource to load only once.
    """
    st.write(f"Loading model '{model_name}' from Hugging Face Hub...")
    try:
        # T5Tokenizer requires the 'sentencepiece' library
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        st.write("Model loaded successfully!")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please make sure the model name is correct and public on Hugging Face.")
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
st.write(f"This app uses a fine-tuned T5 model (`{MODEL_NAME}`) from the Hugging Face Hub to summarize your text.")

# Load the model
model, tokenizer = load_model_and_tokenizer(MODEL_NAME)

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
