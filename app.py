import streamlit as st
from PIL import Image
import easyocr
from pix2tex.cli import LatexOCR
import torch

st.set_page_config(page_title="Handwritten to Text/LaTeX", layout="wide")

st.title("📝 Handwritten Assignment to Editable Text / LaTeX")

uploaded_file = st.file_uploader("Upload handwritten image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    st.write("🔍 Processing...")

    # Initialize OCR
    reader = easyocr.Reader(['en'], gpu=False)
    model = LatexOCR()

    # Extract normal text
    text_results = reader.readtext(image, detail=0)
    normal_text = "\n".join(text_results)

    # Extract math as LaTeX
    latex_result = model(image)

    st.subheader("📄 Extracted Text")
    st.text_area("Editable Text", normal_text, height=200)

    st.subheader("🧮 Extracted LaTeX")
    st.text_area("LaTeX Code", latex_result, height=200)

    st.subheader("🔎 Rendered LaTeX")
    st.latex(latex_result)
