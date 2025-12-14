#!/usr/bin/env python
# coding: utf-8

import os
import io
import json
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import torch

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    pipeline
)

# =========================
# OPTIONAL DEEPFACE IMPORT
# =========================
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Classify images",
    page_icon="📊",
    layout="wide"
)

st.sidebar.header("Classify images")
st.header("AI assistant for image processing")
st.divider()

st.markdown(
    """
    Upload an image and the assistant will:
    - generate a caption
    - detect objects
    - classify the image
    - recognize faces
    - detect emotions
    """
)
st.divider()


# =========================
# BLUE UI STYLE
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
        section[data-testid="stSidebar"] {
            background-color: #081a33;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# UTILS
# =========================
def read_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def img_caption(model, processor, img):
    inputs = processor(img, return_tensors="pt")
    outputs = model.generate(**inputs)
    return processor.decode(outputs[0], skip_special_tokens=True)


def zeroshot(classifier, classes, img):
    return classifier(img, candidate_labels=classes)


# =========================
# LOAD MODELS
# =========================
with st.spinner("Initializing models..."):

    # ---- BLIP image captioning ----
    CAPTION_MODEL_NAME = "Salesforce/blip-image-captioning-base"
    CAPTION_PROCESSOR = BlipProcessor.from_pretrained(CAPTION_MODEL_NAME)
    CAPTION_MODEL = BlipForConditionalGeneration.from_pretrained(
        CAPTION_MODEL_NAME
    )

    # ---- YOLOv5 object detection ----
    try:
        DET_MODEL = torch.hub.load(
            "ultralytics/yolov5",
            "yolov5s",
            pretrained=True
        )
        DET_MODEL.eval()
        YOLO_AVAILABLE = True
    except Exception:
        YOLO_AVAILABLE = False

    # ---- Zero-shot image classification (CLIP) ----
    ZERO_CLASSIFIER = pipeline(
        "zero-shot-image-classification",
        model="openai/clip-vit-base-patch16",
        framework="pt",
        device=-1
    )

    # ---- App config ----
    APP_CONFIG = read_json("config.json")
    CLASSES = APP_CONFIG["classes"]
    DB_DICT = APP_CONFIG["db_dict"]
    TH_OTHERS = APP_CONFIG["th_others"]
    IMGS_PATH = APP_CONFIG["imgs_path"]

    for v in DB_DICT.values():
        os.makedirs(f"{IMGS_PATH}/{v}", exist_ok=True)
    os.makedirs(f"{IMGS_PATH}/other", exist_ok=True)

    DB_PATH = f"{IMGS_PATH}/db"
    os.makedirs(DB_PATH, exist_ok=True)


# =========================
# IMAGE UPLOAD
# =========================
st.subheader("Upload your image")
uploaded_file = st.file_uploader(
    "Select a JPG or PNG image",
    type=["jpg", "png"]
)

if uploaded_file is not None:
    img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    img_np = np.array(img)
    file_name = uploaded_file.name

    with st.spinner("Processing image..."):

        # -------------------------
        # IMAGE CAPTIONING
        # -------------------------
        caption = img_caption(
            CAPTION_MODEL,
            CAPTION_PROCESSOR,
            img
        )
        st.image(img, caption=caption)
        st.write(f"Caption: {caption}")

        # -------------------------
        # OBJECT DETECTION
        # -------------------------
        st.subheader("Detected objects")
        if YOLO_AVAILABLE:
            results = DET_MODEL(img)
            results.render()
            st.image(
                Image.fromarray(results.ims[0]),
                use_container_width=True
            )

            labels = (
                results.pandas()
                .xyxy[0]["name"]
                .unique()
                .tolist()
            )
            st.write(
                ", ".join(labels)
                if labels else "No objects detected"
            )
        else:
            st.warning("Object detection disabled")

        # -------------------------
        # IMAGE CLASSIFICATION
        # -------------------------
        st.subheader("Image classification")
        scores = zeroshot(
            ZERO_CLASSIFIER,
            CLASSES,
            img
        )
        best = max(scores, key=lambda x: x["score"])

        if best["score"] >= TH_OTHERS:
            category = best["label"]
            folder = DB_DICT[category]
        else:
            category = "unknown"
            folder = "other"

        img.save(f"{IMGS_PATH}/{folder}/{file_name}")
        st.write(f"Predicted category: {category}")

        df = pd.DataFrame(scores).set_index("label")
        st.bar_chart(df["score"])

        # -------------------------
        # FACE + EMOTION ANALYSIS
        # -------------------------
        st.subheader("Face and emotion analysis")
        if DEEPFACE_AVAILABLE:
            try:
                emotions = DeepFace.analyze(
                    img_path=img_np,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend="opencv"
                )

                if isinstance(emotions, list):
                    for i, e in enumerate(emotions):
                        st.write(
                            f"Face {i+1}: {e['dominant_emotion']}"
                        )
                else:
                    st.write(
                        f"Emotion: {emotions['dominant_emotion']}"
                    )
            except Exception as e:
                st.warning(f"Emotion analysis failed: {e}")
        else:
            st.warning(
                "Face and emotion analysis disabled "
                "(DeepFace not installed)"
            )
else:
    st.info("Upload an image to start.")
