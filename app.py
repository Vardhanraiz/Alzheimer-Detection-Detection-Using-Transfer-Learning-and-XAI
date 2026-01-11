import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# =========================
# Load trained model
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("alzheimer_model.h5")

model = load_model()

class_names = [
    "Non-Demented",
    "Very Mild Demented",
    "Mild Demented",
    "Moderate Demented"
]

# =========================
# Streamlit UI
# =========================
st.title("Explainable AI-Based Alzheimer’s Disease Detection")
st.write("Upload a Brain MRI image to predict Alzheimer stage and view Grad-CAM explanation.")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "png", "jpeg"]
)

# =========================
# Image preprocessing
# =========================
def preprocess_image(img):
    img = img.convert("RGB")            # force RGB (MRI may be grayscale)
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# Grad-CAM function
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# =========================
# Main logic
# =========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.subheader("Uploaded MRI Image")
    st.image(img, use_column_width=True)

    processed_img = preprocess_image(img)

    # Prediction
    preds = model.predict(processed_img)
    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    st.subheader("Prediction Result")
    st.write("**Alzheimer Stage:**", predicted_class)
    st.write("**Confidence Score:**", round(confidence, 2))

    # Grad-CAM
    heatmap = make_gradcam_heatmap(processed_img, model)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heat*
