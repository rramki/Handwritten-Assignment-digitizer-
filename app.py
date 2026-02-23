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

    # Handwritten recognition using TrOCR
    st.write("🧠 Processing Handwritten Recognition...")
    
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    trocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    model = LatexOCR()
    latex_result = model(image)

    st.subheader("✍ TrOCR Output")
    st.text_area("Handwritten Model Output", trocr_text, height=200)

    st.subheader("🧮 Extracted LaTeX")
    st.text_area("LaTeX Code", latex_result, height=200)

    st.subheader("🔎 Rendered LaTeX")
    st.latex(latex_result)
