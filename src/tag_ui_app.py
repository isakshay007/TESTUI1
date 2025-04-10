import streamlit as st
import joblib
import pickle
import numpy as np
import os
import torch
import types
from torch import nn
from transformers import DistilBertTokenizer, DistilBertModel
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from hmm import HMM_Tagger

# 🩹 Torch patch to avoid Streamlit crash on reload
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
def load_models():
    # HF downloads
    model_path = hf_hub_download(repo_id="iakshay777/stackoverflow-tag-model", filename="trained_model.pt", repo_type="model")
    mlb_path = hf_hub_download(repo_id="iakshay777/stackoverflow-tag-model", filename="mlb.pkl", repo_type="model")

    # Local models
    ml_model = joblib.load("models/tagging_model.pkl")
    mlb_ml = joblib.load("models/tagging_mlb.pkl")

    hmm_model = HMM_Tagger()
    hmm_model.load_model("models/hmm_model.pkl")

    with open(mlb_path, "rb") as f:
        mlb_bert = pickle.load(f)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = MiniTagTransformer(num_tags=len(mlb_bert.classes_))
    bert_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    bert_model.eval()

    return ml_model, mlb_ml, hmm_model, bert_model, mlb_bert, tokenizer

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
st.markdown("""
Welcome to the **StackOverflow AI Tagging System**!  
This tool helps you generate relevant tags for your technical questions using:
- 🧠 Logistic Regression (ML)
- 🔍 Hidden Markov Model (HMM)
- 🤖 DistilBERT Transformer

👇 Select a model and input your question details to get started.
""")

with st.spinner("Loading models..."):
    ml_model, mlb_ml, hmm_model, bert_model, mlb_bert, tokenizer = load_models()

model_choice = st.selectbox("📊 Select a Prediction Model", [
    "Logistic Regression (ML)",
    "Hidden Markov Model (HMM)",
    "DistilBERT Transformer"
])

st.subheader("📝 Enter Your Question")
title = st.text_input("📌 Title", placeholder="e.g., How to merge dictionaries in Python?")
description = st.text_area("🧐 Description", placeholder="Add details, error messages, or examples", height=200)

if st.button("🔍 Generate Tags"):
    if not title.strip() or not description.strip():
        st.warning("Please enter both a title and a description.")
    else:
        with st.spinner("Analyzing..."):
            if model_choice == "Logistic Regression (ML)":
                tags, scores = predict_ml(ml_model, mlb_ml, title, description)
            elif model_choice == "Hidden Markov Model (HMM)":
                tags = predict_hmm(hmm_model, title, description)
            else:
                tags, scores = predict_bert(f"{title} {description}", bert_model, tokenizer, mlb_bert)

        st.subheader("🎯 Predicted Tags")
        if tags:
            if model_choice == "Hidden Markov Model (HMM)":
                for tag, score in tags[:10]:
                    st.write(f"**{tag}**: {score:.3f}")
            else:
                st.write(", ".join(tags))
                st.subheader("📊 Top Scores")
                for tag, score in scores[:10]:
                    st.write(f"**{tag}**: {score:.3f}")
        else:
            st.write("No tags found above threshold.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit · Powered by ML, HMM, and BERT")
