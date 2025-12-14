#!/usr/bin/env python
# coding: utf-8

import streamlit as st
from pathlib import Path
from PIL import Image, UnidentifiedImageError


# ===============================
# CONSTANTS
# ===============================
ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png")
EXCLUDED_FOLDERS = {".ipynb_checkpoints", "db", "rag"}


# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Images gallery",
    page_icon="🖼️",
    layout="wide"
)

st.cache_data.clear()
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
        .stSlider > div {
            color: #e6edf3;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ===============================
# HEADER
# ===============================
st.sidebar.header("Images gallery")

st.header(
    "🖼️ Images database by category"
)

st.markdown(
    """
    Browse all images that were automatically classified
    by the AI assistant. Images are grouped by category
    for easier visual inspection.
    """
)

st.divider()


# ===============================
# BASE PATH (SOURCE OF TRUTH)
# ===============================
IMGS_PATH = Path("/home/jovyan/dlba/topic_18/app/data")


# ===============================
# SAFE IMAGE LOADER
# ===============================
@st.cache_data
def load_images(base_path: Path):
    data = {}

    if not base_path.exists():
        return data

    for folder in sorted(p for p in base_path.iterdir() if p.is_dir()):
        if folder.name in EXCLUDED_FOLDERS:
            continue

        images = []

        for f in folder.iterdir():
            if not f.is_file() or f.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue

            try:
                with Image.open(f) as img:
                    img.verify()

                images.append({
                    "img_name": f.stem,
                    "img_path": str(f)
                })

            except (UnidentifiedImageError, OSError):
                continue

        data[folder.name] = images

    return data


# ===============================
# CONTROLS (NO BLOCK WRAPPER)
# ===============================
n_cols = st.slider(
    "Gallery width (columns)",
    min_value=1,
    max_value=5,
    value=3
)


# ===============================
# DISPLAY GALLERY
# ===============================
imgs_list = load_images(IMGS_PATH)

for category, images in imgs_list.items():
    st.markdown(
        f"""
        <div class="block">
            <h4>Category: {category}</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

    if not images:
        st.info("No images in this category yet.")
        continue

    cols = st.columns(n_cols)

    for i, img in enumerate(images):
        with cols[i % n_cols]:
            st.image(
                img["img_path"],
                caption=img["img_name"],
                use_container_width=True
            )

    st.divider()
