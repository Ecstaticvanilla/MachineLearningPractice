import streamlit as st
import cv2
from PIL import Image
from googletrans import Translator 
import asyncio
import numpy as np
import tensorflow as tf
import time
from sklearn.preprocessing import LabelEncoder


#Loading the model
def load_model():
    try:
        with open('model.json', 'r') as json_file:
            model_json = json_file.read()  
        model = tf.keras.models.model_from_json(model_json)
        model.load_weights('model_weights.weights.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

#prediction function
def predict_model(model, image):
    image = image.reshape(1, 64, 64, 1)  
    image = image.astype('float32') / 255.0 
    prediction = model.predict(image)
    return np.argmax(prediction)  

#This is a function to translate the text
def translate_to_language(word, selected_language):
    if not word:
        return ""

    try:
        translator = Translator()
        result = asyncio.run(translator.translate(word, dest=selected_language))
        return result.text
    except Exception as e:
        return f"Translation error: {e}"


#This is a function to preprocess the image
def imagepreprocessing(image, target_size=(64, 64), minValue=60):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    adaptive_thresh = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )
    _, thresholded = cv2.threshold(
        adaptive_thresh,
        minValue,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    resized = cv2.resize(thresholded, target_size)
    return resized

#This is a function to process live video
def videopreprocessing(frame, target_size=(64, 64), minValue=60):
    # roi = frame[100:356, 100:356]  
    # gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # blur = cv2.GaussianBlur(gray, (5, 5), 2)
    # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # ret, res = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # resized_image = cv2.resize(res, target_size)
    # frame_with_roi = frame.copy()
    # cv2.rectangle(frame_with_roi, (100, 100), (356, 356), (0, 255, 0), 2) 
    # return resized_image, frame_with_roi
    
    x1, y1 = 100, 100
    x2, y2 = 356, 356
    target_size = (64, 64)
    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 2)
    th3 = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    _, res = cv2.threshold(
        th3, minValue, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    resized_image = cv2.resize(res, target_size)
    frame_with_roi = frame.copy()
    cv2.rectangle(frame_with_roi, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return resized_image, frame_with_roi


def map_to_char(prediction):
    if prediction >= 0 and prediction <= 25:
        return chr(prediction + ord('a'))
    else:
        return "Unknown"
 

def main():
    st.title("Machine Learning model for ASL")

    st.sidebar.title("Choose a language")

    selected_language = st.sidebar.selectbox("Select Language", ["Hindi", "Marathi", "Gujarati"])

    tab1,tab2 = st.tabs(["Access using Live Camera","Take photo and upload"])

    # with tab2:
    #     img_file = st.camera_input("Take a picture")

    #     if img_file is not None:
    #         img = Image.open(img_file)
    #         st.image(img, caption="Captured Image")
    #         img = imagepreprocessing(img)
    #         st.image(img, caption="PreProcessed Image")    

    with tab1:
        camera = cv2.VideoCapture(0)

        start_time = time.time()
        tword = ""
        tempfeed = st.empty()
        # tempword = st.empty()
        # translation_interval = 5 
        last_translation_time = time.time()
        while True:
            ret, frame = camera.read()
            if not ret:
                st.error("Failed to retrieve frame from webcam.")
                break
            frame = cv2.flip(frame, 1)
            final_frame, frame_with_roi = videopreprocessing(frame)
            tempfeed.image(frame_with_roi, channels="BGR", use_container_width=True)

            if time.time() - last_translation_time >= 10:
                st.image(final_frame, caption="PreProcessed Image")  
                predicted_class = predict_model(model, final_frame)
                character = map_to_char(predicted_class)
                st.write(f"Predicted: {character}")
                last_translation_time = time.time()

            # if time.time() - start_time >= 5:
                # input_image = np.expand_dims(final_frame, axis=0)
                # prediction = predict_model(final_frame)
                # predicted_label = map_to_char(np.argmax(prediction))
                # tword += predicted_label
        #         start_time = time.time()  

        #     word_placeholder.text("Accumulated Word: " + word)
            # if time.time() - last_translation_time >= translation_interval:
        #         translation = translate_to_language(word, selected_language)
        #         st.write("Translated Text:", translation)
        #         last_translation_time = time.time()
        #         word = ""  
        # camera.release()
        # cv2.destroyAllWindows()            




if __name__ == "__main__":
    main()