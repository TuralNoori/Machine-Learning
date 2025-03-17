import streamlit as st
import numpy as np
import joblib
import matplotlib.pyplot as plt
import cv2
import streamlit_drawable_canvas as draw

MODEL_PATH = "best_model1.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

st.title("Handwritten Digit Recognition")
st.write("Draw a digit between 0 and 9 in the box below and click 'Predict'.")

canvas_result = draw.st_canvas(
    width=280, height=280, drawing_mode="freedraw", key="canvas",
    stroke_width=15, stroke_color="black", background_color="white"
)

if st.button("Predict"):
    if canvas_result.image_data is None:
        st.error("Please draw a digit before predicting.")
    else:
        img = np.array(canvas_result.image_data, dtype=np.uint8)

        if img.shape[-1] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if np.mean(img) > 127:
            img = 255 - img  

        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(contours[0])
            img = img[y:y+h, x:x+w]

        padding = max(10, int(0.2 * max(img.shape))) 
        img = cv2.copyMakeBorder(img, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)

        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img = img.astype('float32') / 255.0

        fig, ax = plt.subplots(1, 2, figsize=(8, 4))
        ax[0].imshow(canvas_result.image_data, cmap="gray") 
        ax[0].set_title("Original Image")
        ax[1].imshow(img, cmap="gray") 
        ax[1].set_title("Processed Image")
        st.pyplot(fig)

        img_flat = img.flatten().reshape(1, -1)

        try:
            prediction = model.predict(img_flat)
            predicted_digit = prediction[0]

            st.write(f"**Model's prediction: {predicted_digit}**")

            probabilities = model.predict_proba(img_flat)[0]  
            st.write("Class Probabilities:", probabilities)

            fig, ax = plt.subplots()
            ax.bar(range(10), probabilities, tick_label=range(10))
            ax.set_xlabel('Digits')
            ax.set_ylabel('Predicted Probability')
            ax.set_title('Predicted Probabilities for Each Digit')
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Something went wrong with the prediction: {e}")