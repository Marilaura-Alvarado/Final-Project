#!/usr/bin/env python
# coding: utf-8

import io
import streamlit as st
import pytesseract
from PIL import Image
from pdf2image import convert_from_bytes
from transformers import pipeline


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Text Extraction & Summarization",
    page_icon="📄",
    layout="wide"
)

# ===============================
# BLUE THEME
# ===============================
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #0b1f3a, #0e2a52);
            color: #e6edf3;
        }
        h1, h2, h3 {
            color: #7fb0ff;
        }
        .block {
            background-color: #0f2f5c;
            border-left: 6px solid #7fb0ff;
            padding: 1.2rem;
            border-radius: 12px;
            margin-bottom: 1.2rem;
        }
        section[data-testid="stSidebar"] {
            background-color: #081a33;
        }
        textarea {
            background-color: #081a33 !important;
            color: #e6edf3 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# HEADER
# ===============================
st.title("📄 OCR & Text Summarization")
st.subheader("Extract text from images or PDFs and summarize it using AI")
st.divider()

st.sidebar.header("Text tools")
st.sidebar.info("Upload an image or PDF to begin")

# ===============================
# LOAD SUMMARIZATION MODEL (SMALL + PYTORCH)
# ===============================
with st.spinner("Loading summarization model..."):
    SUMMARIZATOR = pipeline(
        task="summarization",
        model="Falconsai/text_summarization",
        framework="pt",
        device=-1
    )

# ===============================
# FILE UPLOADER
# ===============================
uploaded_file = st.file_uploader(
    "Upload an image or PDF file",
    type=["png", "jpg", "jpeg", "pdf"]
)

# ===============================
# OCR + SUMMARIZATION PIPELINE
# ===============================
if uploaded_file is not None:
    extracted_text = ""

    # ---------- OCR ----------
    with st.spinner("Extracting text with OCR..."):
        if uploaded_file.type == "application/pdf":
            images = convert_from_bytes(uploaded_file.read())
            for page in images:
                extracted_text += pytesseract.image_to_string(page, lang="eng")
        else:
            image = Image.open(io.BytesIO(uploaded_file.read()))
            extracted_text = pytesseract.image_to_string(image, lang="eng")

    st.markdown('<div class="block">', unsafe_allow_html=True)
    st.subheader("📄 Extracted text")
    st.text_area("OCR output", extracted_text, height=280)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- SUMMARY ----------
    if extracted_text.strip():
        with st.spinner("Generating summary..."):
            summary = SUMMARIZATOR(
                extracted_text[:3000],  # safe limit
                max_length=150,
                min_length=50,
                do_sample=False
            )[0]["summary_text"]

        st.markdown('<div class="block">', unsafe_allow_html=True)
        st.subheader("🧠 Summary")
        st.success(summary)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("No readable text was detected in the file.")
