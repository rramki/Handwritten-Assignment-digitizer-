import streamlit as st
from PIL import Image
import easyocr
import numpy as np
from pix2tex.cli import LatexOCR
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

@st.cache_resource
def load_reader():
    return easyocr.Reader(['en'], gpu=False)
reader = load_reader()


@st.cache_resource
def load_trocr():
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

    model.to("cpu")
    return processor, model
    
processor, model = load_trocr()


st.set_page_config(page_title="Handwritten OCR Tool", layout="wide")
st.title("📝 Handwritten Assignment → Text / LaTeX")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

# Pass numpy array to EasyOCR
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image)

    # Convert to numpy for EasyOCR
    image_np = np.array(image)

    text = reader.readtext(image_np, detail=0)
    extracted_text = "\n".join(text)

    st.text_area("Extracted Text", extracted_text)
    
    # Normal Text OCR
    st.subheader("📄 Extracted Text")
    st.text_area("Editable Text", extracted_text, height=200)


    
