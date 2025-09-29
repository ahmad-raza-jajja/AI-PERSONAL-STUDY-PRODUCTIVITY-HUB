import streamlit as st
from huggingface_hub import InferenceClient

# --- Hugging Face Token from Streamlit Secrets ---
HF_TOKEN = st.secrets["general"]["HF_TOKEN"]

# --- Hugging Face Client ---
client = InferenceClient("sshleifer/distilbart-cnn-12-6", token=HF_TOKEN)

# --- Streamlit Page Config ---
st.set_page_config(
    page_title="AI Study & Productivity Hub",
    page_icon="📚",
    layout="wide",
)

# --- Custom CSS for Sleek Dark + Neon UI ---
st.markdown("""
<style>
body {
    background-color: #0b1120;
    color: #ffffff;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
h1 {
    background: linear-gradient(90deg, #00f5ff, #3b82f6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 3em;
    text-align: center;
    margin-bottom: 5px;
}
h3 {
    color: #94a3b8;
    text-align: center;
    margin-bottom: 30px;
}
.stTextArea textarea {
    background-color: #1e293b;
    color: #ffffff;
    border-radius: 15px;
    padding: 15px;
    font-size: 16px;
}
.stButton>button {
    background: linear-gradient(90deg, #3b82f6, #00f5ff);
    color: #000000;
    font-weight: bold;
    border-radius: 15px;
    padding: 12px 25px;
    font-size: 16px;
    transition: 0.3s;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #00f5ff, #3b82f6);
    color: #000000;
}
.summary-box {
    background: #1e293b;
    padding: 25px;
    border-radius: 20px;
    color: #00f5ff;
    font-size: 18px;
    line-height: 1.8;
    margin-top: 20px;
    box-shadow: 0 0 25px #00f5ff;
}
footer {
    color: #94a3b8;
    text-align: center;
    margin-top: 50px;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>📚 AI Personal Study & Productivity Hub</h1>", unsafe_allow_html=True)
st.markdown("<h3>Boost your study with AI-powered summarization!</h3>", unsafe_allow_html=True)

# --- Text Area ---
text = st.text_area("Paste your text here to generate study notes:", height=300)

# --- Summarize Button ---
if st.button("Summarize"):
    if text.strip():
        with st.spinner("Generating summary..."):
            try:
                summary_result = client.summarization(text)

                # --- Proper extraction of summary_text ---
                summary_text = ""
                if isinstance(summary_result, list) and len(summary_result) > 0:
                    item = summary_result[0]
                    # Hugging Face may return object or dict
                    if hasattr(item, "summary_text"):
                        summary_text = item.summary_text
                    elif isinstance(item, dict) and "summary_text" in item:
                        summary_text = item["summary_text"]
                    elif isinstance(item, dict) and "generated_text" in item:
                        summary_text = item["generated_text"]
                    else:
                        summary_text = str(item)
                else:
                    summary_text = str(summary_result)

                # Display summary
                st.success("✅ Summary Generated!")
                st.markdown(f"<div class='summary-box'>{summary_text}</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("⚠ Please enter some text first!")

# --- Footer ---
#st.markdown("<footer>Developed by Ahmad Raza Jajja</footer>", unsafe_allow_html=True)
