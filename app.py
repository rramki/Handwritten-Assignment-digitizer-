import streamlit as st
from PIL import Image
import easyocr
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

st.set_page_config(page_title="Handwritten OCR Tool", layout="wide")
st.title("📝 Handwritten Assignment → Text / LaTeX")

uploaded_file = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])

@st.cache_resource
def load_models():
    reader = easyocr.Reader(['en'], gpu=False)
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-handwritten")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-small-handwritten")
    return reader, processor, model

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, use_column_width=True)

    reader, processor, model = load_models()

    st.write("🔍 Extracting Text...")
    
    # Normal Text OCR
    text = reader.readtext(image, detail=0)
    extracted_text = "\n".join(text)

    st.subheader("📄 Extracted Text")
    st.text_area("Editable Text", extracted_text, height=200)

    # Handwritten recognition using TrOCR
    st.write("🧠 Processing Handwritten Recognition...")
    
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    trocr_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    st.subheader("✍ TrOCR Output")
    st.text_area("Handwritten Model Output", trocr_text, height=200)
