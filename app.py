import streamlit as st
from huggingface_hub import InferenceClient
import os
from dotenv import load_dotenv

# Load Hugging Face token
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
HF_TOKEN = st.secrets["general"]["HF_TOKEN"]


# Initialize Hugging Face client
client = InferenceClient("facebook/bart-large-cnn", token=HF_TOKEN)

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="AI Study & Productivity Hub",
    page_icon="📚",
    layout="wide",
)

# --- Header ---
st.markdown(
    "<h1 style='text-align:center; color:#00f5ff;'>📚 AI Personal Study & Productivity Hub</h1>",
    unsafe_allow_html=True
)
st.write("Boost your study with AI-powered summarization!")

# --- Text Area ---
text = st.text_area(
    "Paste your text here to generate study notes:", 
    height=250
)

# --- Summarize Button ---
if st.button("Summarize"):
    if text.strip():
        with st.spinner("Generating summary..."):
            try:
                summary_result = client.summarization(text)

                # Safe extraction of summary text
                if isinstance(summary_result, list) and len(summary_result) > 0:
                    if "summary_text" in summary_result[0]:
                        summary_text = summary_result[0]["summary_text"]
                    elif "generated_text" in summary_result[0]:
                        summary_text = summary_result[0]["generated_text"]
                    else:
                        summary_text = str(summary_result[0])
                else:
                    summary_text = str(summary_result)

                # Display summary
                st.success("✅ Summary Generated!")
                st.markdown(
                    f"<div style='padding:20px; background-color:#0f172a; color:white; border-radius:12px; font-size:16px;'>{summary_text}</div>",
                    unsafe_allow_html=True
                )

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("⚠ Please enter some text first!")

# --- Footer ---
st.markdown(
    "<p style='text-align:center; color:#00f5ff; margin-top:40px;'>Developed by Ahmad Raza Jajja</p>",
    unsafe_allow_html=True
)

