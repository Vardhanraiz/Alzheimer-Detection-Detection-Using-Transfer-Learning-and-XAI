import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import sqlite3
import hashlib

from lime import lime_image
from skimage.segmentation import mark_boundaries


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="NeuroScan AI",
    page_icon="🧠",
    layout="wide"
)


# ---------------------------------------------------
# BACKGROUND STYLE
# ---------------------------------------------------

def set_background(image_url):

    st.markdown(
        f"""
        <style>
        .stApp {{
        background-image: url("{image_url}");
        background-size: cover;
        background-position: center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# ---------------------------------------------------
# DATABASE
# ---------------------------------------------------

conn = sqlite3.connect("users.db", check_same_thread=False)
c = conn.cursor()

c.execute(
"""
CREATE TABLE IF NOT EXISTS users(
username TEXT,
password TEXT
)
"""
)

conn.commit()


# ---------------------------------------------------
# PASSWORD HASH
# ---------------------------------------------------

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


# ---------------------------------------------------
# REGISTER USER
# ---------------------------------------------------

def register_user(username, password):

    c.execute(
        "INSERT INTO users VALUES (?,?)",
        (username, hash_password(password))
    )

    conn.commit()


# ---------------------------------------------------
# LOGIN USER
# ---------------------------------------------------

def login_user(username, password):

    c.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, hash_password(password))
    )

    return c.fetchone()


# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


# ---------------------------------------------------
# LOGIN PAGE
# ---------------------------------------------------

def login_page():

    set_background(
        "https://images.unsplash.com/photo-1581091870627-3e7e1b1c0d35"
    )

    st.title("🧠 NeuroScan AI")

    st.subheader("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):

        user = login_user(username, password)

        if user:
            st.session_state.logged_in = True
            st.success("Login Successful")
            st.rerun()

        else:
            st.error("Invalid credentials")


# ---------------------------------------------------
# REGISTER PAGE
# ---------------------------------------------------

def register_page():

    set_background(
        "https://images.unsplash.com/photo-1579154204601-01588f351e67"
    )

    st.title("Create Account")

    username = st.text_input("Create Username")
    password = st.text_input("Create Password", type="password")

    if st.button("Register"):

        register_user(username, password)

        st.success("Account created successfully")


# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

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


# ---------------------------------------------------
# IMAGE PREPROCESS
# ---------------------------------------------------

def preprocess_image(img):

    img = img.convert("RGB")
    img = img.resize((224,224))
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    return img_array


# ---------------------------------------------------
# GRADCAM
# ---------------------------------------------------

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
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]

    heatmap = tf.reduce_sum(pooled_grads * conv_outputs, axis=-1)

    heatmap = tf.maximum(heatmap,0)
    heatmap /= tf.reduce_max(heatmap)+1e-8

    return heatmap.numpy()


# ---------------------------------------------------
# LIME
# ---------------------------------------------------

def lime_predict(images):

    images = np.array(images)/255.0

    return model.predict(images)


def generate_lime_explanation(img):

    explainer = lime_image.LimeImageExplainer()

    img_rgb = img.convert("RGB").resize((224,224))
    img_array = np.array(img_rgb)

    explanation = explainer.explain_instance(
        img_array,
        lime_predict,
        top_labels=1,
        hide_color=0,
        num_samples=300
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )

    lime_result = mark_boundaries(temp/255.0, mask)

    return lime_result


# ---------------------------------------------------
# MAIN APP
# ---------------------------------------------------

def main_app():

    set_background(
        "https://images.unsplash.com/photo-1530023367847-a683933f4172"
    )

    st.sidebar.title("Navigation")

    menu = st.sidebar.radio(
        "Go to",
        ["MRI Analysis","About","Logout"]
    )

    if menu == "Logout":

        st.session_state.logged_in = False
        st.rerun()

    if menu == "About":

        st.title("About NeuroScan AI")

        st.write(
        """
        AI system for Alzheimer's detection using MRI scans.

        Techniques used:

        - Deep Learning
        - Transfer Learning
        - Grad-CAM
        - LIME
        """
        )

    if menu == "MRI Analysis":

        st.title("Alzheimer MRI Analyzer")

        uploaded_file = st.file_uploader(
            "Upload MRI Image",
            type=["jpg","png","jpeg"]
        )

        if uploaded_file:

            img = Image.open(uploaded_file)

            st.image(img)

            if st.button("Analyze MRI"):

                processed = preprocess_image(img)

                preds = model.predict(processed)

                predicted_class = class_names[np.argmax(preds)]
                confidence = float(np.max(preds))

                st.success(f"Prediction: {predicted_class}")
                st.write(f"Confidence: {confidence:.2f}")

                tabs = st.tabs(["GradCAM","LIME"])

                with tabs[0]:

                    heatmap = make_gradcam_heatmap(processed, model)

                    heatmap = cv2.resize(heatmap,(224,224))
                    heatmap = np.uint8(255*heatmap)

                    heatmap = cv2.applyColorMap(
                        heatmap,
                        cv2.COLORMAP_TURBO
                    )

                    st.image(heatmap)

                with tabs[1]:

                    lime_result = generate_lime_explanation(img)

                    st.image(lime_result)


# ---------------------------------------------------
# ROUTING
# ---------------------------------------------------

if st.session_state.logged_in:

    main_app()

else:

    page = st.sidebar.radio(
        "Select",
        ["Login","Register"]
    )

    if page == "Login":

        login_page()

    else:

        register_page()
