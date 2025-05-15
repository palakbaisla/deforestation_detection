import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2

# Set page config
st.set_page_config(page_title="🌳 Forest Segmentation", layout="centered")
st.title("🌳 Forest Segmentation & Deforestation Detection")
# Usage description (added below the title)
st.markdown("""
This app uses deep learning to detect and highlight deforested areas in satellite or drone images.  
It also calculates the percentage of land affected by deforestation for environmental monitoring and research.
""")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("forest_segmentation_unet.h5")
    return model

model = load_model()

# File upload UI
uploaded_file = st.file_uploader("📤 Upload a forest image (JPG, PNG)", type=['jpg', 'jpeg', 'png'])

# Prediction and visualization
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image for model
    IMG_SIZE = 256
    img = np.array(image)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_normalized = img_resized / 255.0  # Normalize if your model used rescale=1./255
    img_input = np.expand_dims(img_normalized, axis=0)

    # Make prediction
    predictions = model.predict(img_input)
    segmentation_mask = np.squeeze(predictions > 0.5).astype(np.uint8)  # Binary mask

    # Convert segmentation mask to standard Python types (int)
    segmentation_mask = segmentation_mask.astype(int)  # Convert to int for serialization

    # Resize mask back to original image size (with NEAREST to keep binary values)
    segmentation_mask_resized = cv2.resize(segmentation_mask, (image.width, image.height), interpolation=cv2.INTER_NEAREST)
    segmentation_mask_resized = segmentation_mask_resized.astype(int)  # Convert to int for serialization

    # Highlight deforested areas (mask == 0) in red
    highlighted_image = np.array(image)
    highlighted_image[segmentation_mask_resized == 0] = [255, 0, 0]  # Red = deforested
    highlighted_image = highlighted_image.astype(int)  # Convert to int for serialization

    # Convert highlighted image and mask to lists for JSON serialization
    segmentation_mask_list = segmentation_mask_resized.tolist()
    highlighted_image_list = highlighted_image.tolist()

    # Display the image with highlighted deforestation
    st.image(highlighted_image, caption="🔍 Highlighted Deforestation", use_container_width=True)

    # Calculate deforestation percentage (mask == 0)
    deforested_area = np.sum(segmentation_mask_resized == 0)
    total_area = segmentation_mask_resized.size
    deforestation_percentage = (deforested_area / total_area) * 100

    # Display result
    st.subheader(f"🌱 Deforestation Detected: **{deforestation_percentage:.2f}%**")

    # Optional debug: show pixel counts
    unique, counts = np.unique(segmentation_mask_resized, return_counts=True)
    # Convert counts to standard Python int types
    counts = counts.astype(int).tolist()  # Convert to list of int for serialization
    st.write("🧪 Mask pixel counts:", dict(zip(unique.tolist(), counts)))

else:
    st.info("Please upload a forest image to check the deforestation.")
