# =========================
# AI Image Caption & Search Engine
# =========================

import streamlit as st
import torch
import clip
import pickle
from PIL import Image
import pandas as pd


# =========================
# Helper Functions
# =========================

# Calculate cosine similarity
def similarity(a, b):
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)

    return (a @ b.T).item()


# Explain matching words
def explain_match(query, caption):
    query_words = set(query.lower().split())
    caption_words = set(caption.lower().split())

    common = query_words.intersection(caption_words)

    return "Common words: " + ", ".join(common) if common else "Semantic match (no direct words)"


# Confidence level
def confidence(score):
    if score > 0.30:
        return "High"

    elif score > 0.25:
        return "Medium"

    else:
        return "Low"


# =========================
# Load CLIP Model
# =========================

@st.cache_resource
def load_model():

    model, preprocess = clip.load("ViT-B/32", device="cpu")
    model.eval()

    return model, preprocess


model, preprocess = load_model()


# =========================
# Load Saved Features
# =========================

@st.cache_data
def load_features():

    with open("image_features.pkl", "rb") as f:
        image_features = pickle.load(f)

    with open("text_features.pkl", "rb") as f:
        text_features = pickle.load(f)

    return image_features, text_features


image_features, text_features = load_features()


# =========================
# Streamlit UI
# =========================

st.title("🚀 AI Image Caption & Search Engine")


# =========================
# Image → Caption
# =========================

st.header("📸 Image → Caption")

# Upload image
uploaded = st.file_uploader("Upload Image")

if uploaded:

    image = Image.open(uploaded).convert("RGB")
    st.image(image)

    st.write("📂 File:", uploaded.name)

    # Generate captions
    if st.button("Top 5 Captions"):

        with st.spinner("Analyzing image..."):

            image_input = preprocess(image).unsqueeze(0)

            # Extract image features
            with torch.no_grad():
                img_feat = model.encode_image(image_input)

            results = []

            # Compare with text features
            for (img, cap), txt_feat in text_features.items():

                score = similarity(img_feat, txt_feat)
                results.append((cap, score))

            # Sort by similarity
            results = sorted(results, key=lambda x: x[1], reverse=True)

        # Best caption
        best_cap, best_score = results[0]

        st.success(f"🏆 Best Match: {best_cap} ({best_score:.2f})")

        st.subheader("Top Matches:")

        # Display top captions
        for cap, score in results[:5]:

            st.write(f"{cap} ({score:.2f}) - {confidence(score)} confidence")
            st.caption(explain_match("", cap))

        # Display graph
        scores_df = pd.DataFrame(results[:5], columns=["Caption", "Score"])

        st.bar_chart(scores_df.set_index("Caption"))


# =========================
# Text → Image Search
# =========================

st.header("🔍 Text → Image Search")

# Input search query
query = st.text_input("Search images")

if st.button("Search"):

    if query.strip() == "":
        st.warning("Please enter a search query")

    else:

        with st.spinner("Searching images..."):

            tokens = clip.tokenize([query])

            # Extract text features
            with torch.no_grad():
                txt_feat = model.encode_text(tokens)

            results = []

            # Compare with image features
            for img, img_feat in image_features.items():

                score = similarity(img_feat, txt_feat)
                results.append((img, score))

            # Sort by similarity
            results = sorted(results, key=lambda x: x[1], reverse=True)

        st.subheader("Top Matching Images:")

        # Display top images
        for img, score in results[:5]:

            st.image(
                f"Images/{img}",
                caption=f"{score:.2f} - {confidence(score)} confidence"
            )


# =========================
# System Explanation
# =========================

st.header("🧠 How it works")

st.write("""

This system uses a pretrained model to understand both images and text.

• Images and captions are converted into embeddings (vectors)

• These embeddings capture semantic meaning (objects, actions, scenes)

• Cosine similarity is used to compare them

• The system returns the most similar matches

""")


# =========================
# Example Queries
# =========================

st.header("💡 Try These Examples")

st.write("""

- a girl playing in water

- a dog running

- children playing

- man riding bicycle

""")