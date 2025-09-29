import streamlit as st
from huggingface_hub import InferenceClient
import cohere
from hashlib import sha256
import os

# --- Page Config ---
st.set_page_config(page_title="AI Study & Productivity Hub", page_icon="📚", layout="wide")

# --- Custom CSS for Neon Dark UI ---
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
.session-history {background:#111827;padding:15px;border-radius:15px;color:#f0f0f0; margin-top:15px; max-height:200px; overflow-y:auto;}
.copy-btn {margin-left:10px;}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("<h1>📚 AI Personal Study & Productivity Hub</h1>", unsafe_allow_html=True)
st.markdown("<h3>Boost your study with AI-powered tools!</h3>", unsafe_allow_html=True)

# --- Session State ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'username' not in st.session_state: st.session_state.username = ""
if 'history' not in st.session_state: st.session_state.history = []

# --- Password Hashing ---
def hash_password(pw): return sha256(pw.encode()).hexdigest()

# --- Simple Auth ---
users = {}  # demo only
def signup(username, password):
    if username in users: st.error("Username exists!")
    else: users[username] = hash_password(password); st.success("Account created! Please log in.")
def login(username, password):
    if username in users and users[username]==hash_password(password):
        st.session_state.logged_in=True; st.session_state.username=username
        st.success(f"Welcome {username}!")
    else: st.error("Invalid credentials!")

if not st.session_state.logged_in:
    auth_option = st.radio("Choose Action:", ["Sign In", "Sign Up"])
    uname = st.text_input("Username")
    pw = st.text_input("Password", type="password")
    if st.button("Submit"):
        if auth_option=="Sign Up": signup(uname,pw)
        else: login(uname,pw)
else:
    st.markdown(f"### Logged in as: {st.session_state.username}")

    # --- API Clients ---
    HF_TOKEN = st.secrets["general"]["HF_TOKEN"]
    COHERE_API_KEY = st.secrets["general"]["COHERE_API_KEY"]

    hf_client = InferenceClient("sshleifer/distilbart-cnn-12-6", token=HF_TOKEN)
    co = cohere.Client(COHERE_API_KEY)

    # --- Text Input ---
    text = st.text_area("Enter your text:", height=250)
    tool = st.selectbox("Select Tool:", ["Summarization","Keyword Extraction","Paraphrasing","Translation"])

    # --- Run Tool ---
    if st.button("Run"):
        if text.strip():
            with st.spinner("Processing..."):
                try:
                    output = ""
                    if tool=="Summarization":
                        # Try HF first, then Cohere fallback
                        try:
                            res = hf_client.summarization(text)
                            if isinstance(res,list) and len(res)>0:
                                output=res[0].get("summary_text") or res[0].get("generated_text") or str(res[0])
                            else: output=str(res)
                        except:
                            response = co.summarize(text)
                            output = response.summary

                    elif tool=="Keyword Extraction":
                        prompt=f"Extract keywords from this text:\n{text}\nKeywords:"
                        response = co.generate(model='xlarge', prompt=prompt, max_tokens=60)
                        output = response.generations[0].text.strip()

                    elif tool=="Paraphrasing":
                        prompt=f"Paraphrase this text:\n{text}\nParaphrased:"
                        response = co.generate(model='xlarge', prompt=prompt, max_tokens=200)
                        output = response.generations[0].text.strip()

                    elif tool=="Translation":
                        prompt=f"Translate this text to Spanish:\n{text}\nTranslation:"
                        response = co.generate(model='xlarge', prompt=prompt, max_tokens=200)
                        output = response.generations[0].text.strip()

                    # --- Display & Copy ---
                    st.success(f"✅ {tool} Result:")
                    st.markdown(f"<div class='summary-box'>{output}</div>",unsafe_allow_html=True)
                    st.download_button("📋 Copy / Download Result", data=output, file_name=f"{tool}_output.txt")

                    # --- Save to session history ---
                    st.session_state.history.append(f"{tool}: {output}")

                except Exception as e: st.error(f"Error: {e}")
        else: st.warning("⚠ Please enter some text!")

    # --- Session History ---
    if st.session_state.history:
        st.markdown("<h3>📝 Session History</h3>", unsafe_allow_html=True)
        for item in st.session_state.history[::-1]:
            st.markdown(f"<div class='session-history'>{item}</div>", unsafe_allow_html=True)

# --- Footer ---
st.markdown("<footer>Developed by Ahmad Raza Jajja</footer>", unsafe_allow_html=True)
