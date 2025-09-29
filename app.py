import streamlit as st
from huggingface_hub import InferenceClient

# --- Page Config ---
st.set_page_config(page_title="AI Study & Productivity Hub", page_icon="📚", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
body {background-color:#0b1120;color:#fff;font-family:'Segoe UI', Tahoma, Geneva, Verdana;}
h1 {background:linear-gradient(90deg,#00f5ff,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent;text-align:center;font-size:3em;margin-bottom:5px;}
h3 {color:#94a3b8;text-align:center;margin-bottom:30px;}
.stTextArea textarea {background-color:#1e293b;color:#fff;border-radius:15px;padding:15px;font-size:16px;}
.stButton>button {background:linear-gradient(90deg,#3b82f6,#00f5ff);color:#000;font-weight:bold;border-radius:15px;padding:12px 25px;font-size:16px;transition:0.3s;}
.stButton>button:hover {background:linear-gradient(90deg,#00f5ff,#3b82f6);color:#000;}
.summary-box {background:#1e293b;padding:25px;border-radius:20px;color:#00f5ff;font-size:18px;line-height:1.8;margin-top:20px;box-shadow:0 0 25px #00f5ff;}
footer {color:#94a3b8;text-align:center;margin-top:50px;}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>📚 AI Personal Study & Productivity Hub</h1>", unsafe_allow_html=True)
st.markdown("<h3>Boost your study with AI-powered summarization!</h3>", unsafe_allow_html=True)

# --- Hugging Face Client ---
HF_TOKEN = st.secrets["general"]["HF_TOKEN"]
client = InferenceClient("sshleifer/distilbart-cnn-12-6", token=HF_TOKEN)

# --- Text Input ---
text = st.text_area("Enter your text here:", height=250)

# --- Summarize ---
if st.button("Summarize"):
    if text.strip():
        with st.spinner("Generating summary..."):
            try:
                result = client.summarization(text)

                # Extract clean text
                if isinstance(result, list) and len(result) > 0:
                    summary_obj = result[0]
                    # Check if summary_text key exists
                    if hasattr(summary_obj, 'summary_text'):
                        summary = summary_obj.summary_text
                    else:
                        summary = str(summary_obj)
                else:
                    summary = str(result)

                # Display summary
                st.success("✅ Summary Generated!")
                st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("⚠ Please enter some text!")

# --- Footer ---
st.markdown("<footer>Developed by Ahmad Raza Jajja</footer>", unsafe_allow_html=True)
