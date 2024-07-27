import streamlit as st
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration

st.set_page_config(
    page_title="Image Captioning App",
    page_icon="📸",
)

st.markdown("""
    <style>
    body {
        background-color: #f0f2f6;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        color: #2c3e50;
        margin-bottom: 0.5em;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        font-size: 1.25em;
        color: #34495e;
        margin-top: -0.5em;
        margin-bottom: 1em;
        font-weight: 300;
    }
    .caption-title {
        font-size: 1.5em;
        color: #2c3e50;
        font-weight: 600;
        text-align: center;
    }
    .caption {
        font-size: 1.2em;
        color: #000;
        margin-top: 0.5em;
        font-weight: 200;
        text-align: left;
        display:flex;
    }
    .captions{
        color: #36454F;
        font-family: cursive;
        font-weight: 800;
        text-align: left;
    }
    .uploaded-image {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 1em;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: grey;
        text-align: center;
        padding: 6px 0;
        z-index:100;
    }
    .stFileUpload {
        color: #fff000;
        background-color: #2c3e50;
        border-radius: 10px;
        padding: 0.5rem;
        text-align: center;
        font-weight: 600;
    }
    .stFileUpload:hover {
        background-color: #34495e;
    }
    </style>
    <div class="footer">
        <p>Developed by Varalakshmi</p>
    </div>
""", unsafe_allow_html=True)

# Set up Streamlit
st.markdown('<div class="title">Image Caption Generator 📸</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image and get generated captions.</div>', unsafe_allow_html=True)

# Upload an image
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Initialize the image captioning model
@st.cache_resource(show_spinner=False)
def load_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

processor, model, device = load_model()

def generate_captions(image):
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    inputs = processor(images=image, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, num_beams=5, num_return_sequences=3, max_length=16, min_length=5)
    captions = [processor.decode(output, skip_special_tokens=True) for output in outputs]
    return captions

# Main content
if uploaded_image is not None:
    # Display uploaded image with reduced size
    image = Image.open(uploaded_image)
    max_image_size = (200, 200)  # Maximum dimensions for the displayed image
    image.thumbnail(max_image_size)  # Reduce image size while maintaining aspect ratio

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Show a spinner while generating the caption
    with st.spinner('Generating captions...'):
        # Generate captions
        captions = generate_captions(image)

    # Display the generated captions
    st.markdown('<div class="caption-container">', unsafe_allow_html=True)
    st.markdown('<div class="caption-title">Generated Captions:</div>', unsafe_allow_html=True)
    for i, caption in enumerate(captions):
        # Capitalize the first word of the caption
        capitalized_caption = caption.capitalize()
        st.markdown(f'<div class="caption">Caption {i+1}: <div class="captions">{capitalized_caption}</div></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)