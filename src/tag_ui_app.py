# 📁 src/tag_ui_app.py
import streamlit as st
import joblib
import os
import torch
import pickle
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn
from huggingface_hub import hf_hub_download
from hmm import HMM_Tagger

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

def load_models():
    BASE_DIR = os.path.dirname(os.path.dirname(__file__))
    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Hugging Face hub downloads
    model_path = hf_hub_download(repo_id="iakshay777/stackoverflow-tag-model", filename="trained_model.pt", repo_type="model")
    mlb_path = hf_hub_download(repo_id="iakshay777/stackoverflow-tag-model", filename="mlb.pkl", repo_type="model")

    ml_model = joblib.load(os.path.join(model_dir, "tagging_model.pkl"))
    mlb_ml = joblib.load(os.path.join(model_dir, "tagging_mlb.pkl"))

    hmm_model = HMM_Tagger()
    hmm_model.load_model(os.path.join(model_dir, "hmm_model.pkl"))

    with open(mlb_path, "rb") as f:
        mlb_bert = pickle.load(f)

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = MiniTagTransformer(num_tags=len(mlb_bert.classes_))
    bert_model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    bert_model.eval()

    return ml_model, mlb_ml, hmm_model, bert_model, mlb_bert, tokenizer

def predict_ml(model, mlb, title, description, threshold=0.08):
    combined_text = title + " " + description
    probs = model.predict_proba([combined_text])[0]
    prob_dict = dict(zip(mlb.classes_, probs))
    sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    predicted_labels = [label for label, score in sorted_probs if score >= threshold]
    return predicted_labels, sorted_probs

def predict_hmm(hmm_model, title, description, threshold=0.1):
    combined_text = title + " " + description
    predicted_tags = hmm_model.predict(combined_text)

    input_sentence = preprocess(description)
    predicted_tags = list(set([preprocess(tag) for tag in predicted_tags]))
    all_text = [input_sentence] + predicted_tags

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_text)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    filtered_tags = [(tag, score) for tag, score in zip(predicted_tags, cosine_similarities) if score >= threshold]
    sorted_tags = sorted(filtered_tags, key=lambda x: x[1], reverse=True)
    return sorted_tags

def predict_bert(text, model, tokenizer, mlb, threshold=0.05, show_top_k=5, fallback=True):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    with torch.no_grad():
        logits = model(inputs["input_ids"], inputs["attention_mask"])
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    top_probs = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:show_top_k]
    predicted_indices = np.where(probs >= threshold)[0]
    tags = [mlb.classes_[i] for i in predicted_indices]

    if fallback and not tags:
        tags = [mlb.classes_[i] for i, _ in top_probs]

    return tags, [(mlb.classes_[i], p) for i, p in top_probs]

# ========= STREAMLIT UI =========
st.set_page_config(page_title="StackOverflow Tag Generator", layout="wide")

st.title("🚀 StackOverflow Tag Generator")
st.markdown("""
Welcome to the **StackOverflow AI Tagging System**!  
This tool helps you generate relevant tags for your technical questions using:
- Logistic Regression (ML)
- Hidden Markov Model (HMM)
- DistilBERT Transformer

👇 Start by selecting the model you'd like to use.
""")

with st.spinner("🔄 Loading models..."):
    ml_model, mlb_ml, hmm_model, bert_model, mlb_bert, tokenizer = load_models()

model_choice = st.selectbox("📊 Select Tag Prediction Model", [
    "Logistic Regression (ML)",
    "Hidden Markov Model (HMM)",
    "DistilBERT Transformer"
])

st.subheader("📝 Provide your Question Details")
title = st.text_input("📌 Question Title", placeholder="e.g., How to merge dictionaries in Python?")
description = st.text_area("🧐 Question Description", placeholder="Provide details about your issue, approach, error, etc.", height=200)

if st.button("Generate Tags"):
    if not title.strip() or not description.strip():
        st.warning("Please fill in both title and description.")
    else:
        with st.spinner("⚙️ Generating tags..."):
            if model_choice == "Logistic Regression (ML)":
                tags, scores = predict_ml(ml_model, mlb_ml, title, description)
                st.subheader("🎯 Predicted Tags")
                st.write(", ".join(tags) if tags else "No tags above threshold.")

                st.subheader("📊 Tag Probabilities")
                for tag, score in scores[:10]:
                    st.write(f"**{tag}**: {score:.3f}")

            elif model_choice == "Hidden Markov Model (HMM)":
                hmm_results = predict_hmm(hmm_model, title, description)
                st.subheader("🎯 Predicted Tags")
                if hmm_results:
                    for tag, score in hmm_results[:10]:
                        st.write(f"**{tag}**: {score:.3f}")
                else:
                    st.write("No relevant tags found.")

            else:
                full_text = title + " " + description
                tags, scores = predict_bert(full_text, bert_model, tokenizer, mlb_bert)
                st.subheader("🎯 Predicted Tags")
                st.write(", ".join(tags))
                st.subheader("📊 Top Tag Probabilities")
                for tag, prob in scores:
                    st.write(f"**{tag}**: {prob:.3f}")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit | Powered by ML, HMM, and BERT")