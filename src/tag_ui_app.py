import streamlit as st
import joblib
import pickle
import numpy as np
import torch
import types
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from hmm import HMM_Tagger

# 🩹 Patch to prevent Streamlit+PyTorch reload crash
if not hasattr(torch.classes, '__path__'):
    torch.classes.__path__ = types.SimpleNamespace(_path=[])

# ========= MODEL CLASSES =========
class MiniTagTransformer(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_tags)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        return self.classifier(cls_output)

# ========= HELPERS =========
def preprocess(text):
    return text.lower().strip()

@st.cache_resource(show_spinner=False)
def load_ml():
    model = joblib.load("models/tagging_model.pkl")
    mlb = joblib.load("models/tagging_mlb.pkl")
    return model, mlb

@st.cache_resource(show_spinner=False)
def load_hmm():
    model = HMM_Tagger()
    model.load_model("models/hmm_model.pkl")
    return model

@st.cache_resource(show_spinner=False)
def load_bert():
    model_path = hf_hub_download(repo_id="iakshay777/stackoverflow-tag-model", filename="trained_model.pt", repo_type="model")
    mlb_path = hf_hub_download(repo_id="iakshay777/stackoverflow-tag-model", filename="mlb.pkl", repo_type="model")
    with open(mlb_path, "rb") as f:
        mlb = pickle.load(f)
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = MiniTagTransformer(num_tags=len(mlb.classes_))
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model, mlb, tokenizer

def predict_ml(model, mlb, title, description, threshold=0.08):
    text = f"{title} {description}"
    probs = model.predict_proba([text])[0]
    sorted_probs = sorted(zip(mlb.classes_, probs), key=lambda x: x[1], reverse=True)
    tags = [tag for tag, score in sorted_probs if score >= threshold]
    return tags, sorted_probs

def predict_hmm(model, title, description, threshold=0.1):
    text = f"{title} {description}"
    predicted = list(set(preprocess(tag) for tag in model.predict(text)))
    tfidf_matrix = TfidfVectorizer().fit_transform([description] + predicted)
    sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    filtered = [(tag, sim) for tag, sim in zip(predicted, sims) if sim >= threshold]
    return sorted(filtered, key=lambda x: x[1], reverse=True)

def predict_bert(text, model, tokenizer, mlb, threshold=0.05, show_top_k=5, fallback=True):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    top_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:show_top_k]
    indices = np.where(probs >= threshold)[0]
    tags = [mlb.classes_[i] for i in indices]
    if fallback and not tags:
        tags = [mlb.classes_[i] for i, _ in top_probs]
    return tags, [(mlb.classes_[i], p) for i, p in top_probs]

# ========= STREAMLIT UI =========
st.set_page_config(page_title="StackOverflow Tag Generator", layout="wide")

st.title("🚀 StackOverflow Tag Generator")

model_choice = st.selectbox("📊 Choose a Tag Prediction Model", [
    "Logistic Regression (ML)",
    "Hidden Markov Model (HMM)",
    "DistilBERT Transformer"
])

if st.button("✅ Select Model"):
    st.session_state.model_selected = model_choice
    if model_choice == "Logistic Regression (ML)":
        st.session_state.ml_model, st.session_state.mlb_ml = load_ml()
    elif model_choice == "Hidden Markov Model (HMM)":
        st.session_state.hmm_model = load_hmm()
    elif model_choice == "DistilBERT Transformer":
        st.session_state.bert_model, st.session_state.mlb_bert, st.session_state.tokenizer = load_bert()

if "model_selected" in st.session_state:
    st.subheader(f"📝 Enter Question for {st.session_state.model_selected}")
    title = st.text_input("📌 Title", placeholder="e.g., How to merge dictionaries in Python?")
    description = st.text_area("🧐 Description", height=200, placeholder="Include details, errors, etc.")

    if st.button("🔍 Generate Tags"):
        if not title.strip() or not description.strip():
            st.warning("Please provide both title and description.")
        else:
            with st.spinner("Generating tags..."):
                if st.session_state.model_selected == "Logistic Regression (ML)":
                    tags, scores = predict_ml(st.session_state.ml_model, st.session_state.mlb_ml, title, description)
                    st.subheader("🎯 Tags")
                    st.write(", ".join(tags) if tags else "No tags found.")
                    st.subheader("📊 Probabilities")
                    for tag, score in scores[:10]:
                        st.write(f"**{tag}**: {score:.3f}")

                elif st.session_state.model_selected == "Hidden Markov Model (HMM)":
                    results = predict_hmm(st.session_state.hmm_model, title, description)
                    st.subheader("🎯 Tags")
                    for tag, score in results[:10]:
                        st.write(f"**{tag}**: {score:.3f}")

                else:
                    tags, scores = predict_bert(
                        f"{title} {description}",
                        st.session_state.bert_model,
                        st.session_state.tokenizer,
                        st.session_state.mlb_bert
                    )
                    st.subheader("🎯 Tags")
                    st.write(", ".join(tags) if tags else "No tags found.")
                    st.subheader("📊 Top Scores")
                    for tag, prob in scores:
                        st.write(f"**{tag}**: {prob:.3f}")

st.markdown("---")
st.caption("Built with ❤️ using Streamlit, Transformers, and scikit-learn")