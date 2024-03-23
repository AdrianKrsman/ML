import streamlit as st
from PIL import Image, ImageOps
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from streamlit_drawable_canvas import st_canvas
from joblib import load, dump

#import SVC model and pipeline
clf = load(r"C:\Users\adria\ec_utbildning\Machine Learning\Ml models\svm_final.joblib")
pipe = load(r"C:\Users\adria\ec_utbildning\Machine Learning\Ml models\preprocess_pipe.joblib")


#create function for formating pictures that are drawn in the canvas
def preprocess_image(image_data):
    img = Image.fromarray((image_data[:, :, :3]).astype(np.uint8))
    img = img.resize((28, 28), Image.LANCZOS)
    img = ImageOps.grayscale(img)
    img_array = np.array(img)
    img_flattened = img_array.flatten()
    img_reshaped = img_flattened.reshape(1, -1)
    img_ready = pipe.transform(img_reshaped)
    return img_ready


#create page with title, instructions and a canvas
st.title("Classifier Model")
st.write("""Draw a digit in the canvas below and the model will predict what digit you have drawn,
             Try to keep the digit upright and in the center of the canvas for the best results.""")
#create a drawable canvas
canvas_result = st_canvas(
    fill_color="black",
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas",
 )
#process the canvas and predict
if canvas_result.image_data is not None:
    preprocessed_image = preprocess_image(canvas_result.image_data)
    prediction = clf.predict(preprocessed_image)
    st.write(f'The predicted digit is: {prediction[0]}')
#create checkbox for user to see the preprocessing code
show_preprocess_code = st.checkbox("Show preprocessing code")
if show_preprocess_code:
    code = """
def preprocess_image(image_data):
    img = Image.fromarray((image_data[:, :, :3]).astype(np.uint8))
    img = img.resize((28, 28), Image.LANCZOS)
    img = ImageOps.grayscale(img)
    img_array = np.array(img)
    img_flattened = img_array.flatten()
    img_reshaped = img_flattened.reshape(1, -1)
    img_ready = pipe.transform(img_reshaped)
    return img_ready
        """
    st.code(code, language='python')
