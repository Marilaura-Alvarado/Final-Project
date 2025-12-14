#!/usr/bin/env python
# coding: utf-8

import os
import io
import json
import datetime
import streamlit as st
from PIL import Image


# ===============================
# CONSTANTS
# ===============================
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Friends database",
    page_icon="🧑‍🤝‍🧑",
    layout="wide"
)


# ===============================
# BLUE THEME
# ===============================
st.markdown(
    """
    <style>
        .main {
            background: linear-gradient(180deg, #0b1f3a 0%, #0e2a52 100%);
            color: #e6edf3;
        }

        h1, h2, h3, h4 {
            color: #7fb0ff;
        }

        section[data-testid="stSidebar"] {
            background-color: #081a33;
        }

        .block {
            background-color: #0f2f5c;
            border-left: 5px solid #7fb0ff;
            padding: 1.2rem;
            border-radius: 14px;
            margin-bottom: 1.5rem;
        }

        textarea, input {
            background-color: #081a33 !important;
            color: #e6edf3 !important;
            border-radius: 8px;
        }

        .stButton > button {
            background-color: #1f4fd8;
            color: white;
            border-radius: 8px;
            font-weight: 600;
        }

        .stButton > button:hover {
            background-color: #163fa8;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ===============================
# HEADER
# ===============================
st.sidebar.header("Friends database")

st.header(
    "🧑‍🤝‍🧑 Friends image database",
    divider="rainbow"
)

st.markdown(
    """
    Browse all stored images of friends and update
    the database by uploading new people.
    """
)

st.divider()


# ===============================
# UTILS
# ===============================
def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_filename(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")


# ===============================
# LOAD CONFIG & PATHS
# ===============================
APP_CONFIG = read_json("config.json")

IMGS_PATH = APP_CONFIG["imgs_path"]
DB_PATH = os.path.join(IMGS_PATH, "db")
FRIENDS_PATH = os.path.join(DB_PATH, "people")

os.makedirs(FRIENDS_PATH, exist_ok=True)


# ===============================
# LOAD FRIENDS IMAGES
# ===============================
@st.cache_data
def load_friends(path):
    if not os.path.exists(path):
        return []

    files = [
        f for f in os.listdir(path)
        if f.lower().endswith(ALLOWED_EXTENSIONS)
    ]

    return [
        {
            "name": os.path.splitext(f)[0],
            "path": os.path.join(path, f)
        }
        for f in sorted(files)
    ]


# ===============================
# GALLERY
# ===============================
st.markdown('<div class="block">', unsafe_allow_html=True)
st.subheader("📸 Friends gallery")

n_cols = st.slider(
    "Gallery width (columns)",
    min_value=1,
    max_value=5,
    value=3
)
st.markdown('</div>', unsafe_allow_html=True)

friends = load_friends(FRIENDS_PATH)

if not friends:
    st.info("No friends in the database yet.")
else:
    cols = st.columns(n_cols)
    for i, img in enumerate(friends):
        with cols[i % n_cols]:
            st.image(
                img["path"],
                caption=img["name"],
                use_container_width=True
            )

st.divider()


# ===============================
# UPLOAD SECTION
# ===============================
st.markdown('<div class="block">', unsafe_allow_html=True)
st.subheader("➕ Add new friend")

uploaded_file = st.file_uploader(
    "Upload an image (JPG, JPEG, PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    filename = clean_filename(uploaded_file.name)
    ext = os.path.splitext(filename)[1].lower()

    if ext in ALLOWED_EXTENSIONS:
        img = Image.open(io.BytesIO(uploaded_file.read()))
        save_path = os.path.join(FRIENDS_PATH, filename)
        img.save(save_path)

        with open("history.log", "a") as log:
            log.write(
                f"{datetime.datetime.now()} - "
                f'friend image "{filename}" added\n'
            )

        st.success("Image successfully added to Friends database.")
        st.cache_data.clear()
        st.rerun()
    else:
        st.error("Invalid file format.")

st.markdown('</div>', unsafe_allow_html=True)
