#!/usr/bin/env python
# coding: utf-8

import os
import json
import streamlit as st


def read_json(file_path):
    with open(file_path) as file:
        return json.load(file)


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Image Assistant",
    page_icon="🔍",
    layout="wide"
)

# =========================
# GLOBAL BLUE STYLE (ONCE)
# =========================
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(
                180deg,
                #0b1f3a 0%,
                #0e2a52 40%,
                #0b1f3a 100%
            );
            color: #e6edf3;
        }

        h1, h2, h3 {
            color: #7fb0ff;
        }

        .card {
            background-color: #0f2f5c;
            border: 1px solid #1c4a8c;
            border-left: 5px solid #7fb0ff;
            padding: 1rem 1.2rem;
            border-radius: 14px;
            margin-bottom: 1rem;
            max-width: 900px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
        }

        .card-title {
            font-weight: 600;
            font-size: 1.05rem;
            margin-bottom: 0.3rem;
            color: #ffffff;
        }

        .card-text {
            color: #c7d8f5;
            font-size: 0.95rem;
        }

        .info-box {
            background-color: #0f2f5c;
            border: 1px dashed #7fb0ff;
            padding: 1rem;
            border-radius: 12px;
            max-width: 900px;
            color: #c7d8f5;
        }

        section[data-testid="stSidebar"] {
            background-color: #081a33;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# HEADER
# =========================
st.title("AI Image Assistant")
st.subheader("Your smart assistant for image understanding")
st.divider()

st.sidebar.success("Select a tool from the menu to get started")

# =========================
# INTRO
# =========================
st.markdown(
    """
    **AI Image Assistant** is an application designed to help you
    analyze, organize, and understand images using modern
    computer vision and language models.
    """
)

# =========================
# FEATURES
# =========================
st.subheader("What can this application do?")

st.markdown(
    """
    <div class="card">
        <div class="card-title">🖼️ Image understanding</div>
        <div class="card-text">
            Automatically generate captions and descriptions for images.
        </div>
    </div>

    <div class="card">
        <div class="card-title">🎯 Object and face detection</div>
        <div class="card-text">
            Detect objects, recognize people, and analyze facial emotions.
        </div>
    </div>

    <div class="card">
        <div class="card-title">🗂️ Smart categorization</div>
        <div class="card-text">
            Classify images into meaningful categories and store them in a database.
        </div>
    </div>

    <div class="card">
        <div class="card-title">📄 Text extraction</div>
        <div class="card-text">
            Extract and summarize text from images and PDF documents.
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# =========================
# HOW IT WORKS
# =========================
st.subheader("How does it work?")

st.markdown(
    """
    <div class="info-box">
        • Multiple computer vision and NLP models work together under the hood.<br>
        • Images and documents are stored in structured folders for easy retrieval.<br>
        • A user-friendly interface allows you to interact with AI tools intuitively.
    </div>
    """,
    unsafe_allow_html=True
)