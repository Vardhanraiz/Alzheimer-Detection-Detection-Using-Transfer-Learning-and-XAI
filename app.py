import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

from lime import lime_image
from skimage.segmentation import mark_boundaries

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
st.write("Upload a Brain MRI image to predict Alzheimer stage and view explanations.")

uploaded_file = st.file_uploader(
    "Choose an MRI image",
    type=["jpg", "png", "jpeg"]
)

# =========================
# Image preprocessing
# =========================
def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# =========================
# Grad-CAM (Cloud-safe)
# =========================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="top_conv"):
    last_conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[last_conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_channel = tf.reduce_max(predictions, axis=1)

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()

# =========================
# LIME helper functions
# =========================
def lime_predict(images):
    images = np.array(images) / 255.0
    return model.predict(images)

def generate_lime_explanation(img):
    explainer = lime_image.LimeImageExplainer()

    img_rgb = img.convert("RGB").resize((224, 224))
    img_array = np.array(img_rgb)

    explanation = explainer.explain_instance(
        img_array,
        lime_predict,
        top_labels=1,
        hide_color=0,
        num_samples=500
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    lime_result = mark_boundaries(temp / 255.0, mask)
    return lime_result

# =========================
# Main logic
# =========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.subheader("Uploaded MRI Image")
    st.image(img, use_column_width=True)

    processed_img = preprocess_image(img)

    preds = model.predict(processed_img)
    predicted_class = class_names[np.argmax(preds)]
    confidence = float(np.max(preds))

    st.subheader("Prediction Result")
    st.write("**Alzheimer Stage:**", predicted_class)
    st.write("**Confidence Score:**", round(confidence, 2))
    st.write("### Class Probabilities")
    for i, cls in enumerate(class_names):
    st.write(f"{cls}: {preds[0][i]:.2f}")


    # -------- Grad-CAM --------
    st.subheader("Grad-CAM Visualization")
    heatmap = make_gradcam_heatmap(processed_img, model)

    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    img_rgb = img.convert("RGB").resize((224, 224))
    img_array = np.array(img_rgb)

    superimposed_img = heatmap * 0.4 + img_array
    superimposed_img = np.uint8(superimposed_img / np.max(superimposed_img) * 255)

    st.image(superimposed_img, use_column_width=True)

    # -------- LIME --------
    st.subheader("LIME Explanation")
    lime_result = generate_lime_explanation(img)
    st.image(lime_result, use_column_width=True)

