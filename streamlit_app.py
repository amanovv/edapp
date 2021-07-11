import streamlit as st
import os
import urllib
import tensorflow as tf
import pandas as pd
import cv2


import numpy as np


def download_file(file_path):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                        (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()

EXTERNAL_DEPENDENCIES = {
    "emotion.model": {
    "url": "https://www.dropbox.com/s/072b5vf4b33bu1l/emotion_detection_model_for_streamlit.h5",
    "size": 126265696
    }
}

EMOTIONS = ['ANGRY', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']

# Download the the model file
download_file("https://www.dropbox.com/s/072b5vf4b33bu1l/emotion_detection_model_for_streamlit.h5")

st.title("Emotion Detector")
st.header("This app detects your emotions! upload a picture to try it out!")

model = tf.keras.models.load_model("emotion_detection_model_for_streamlit.h5")
#model_lm = tf.keras.models.load_model("best_lm_model.h5")
f = st.file_uploader("Upload Image")

if f is not None: 
  # Extract and display the image
  file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
  image = cv2.imdecode(file_bytes, 1)
  st.image(image, channels="BGR")

  # Prepare the image
  resized = cv2.resize(image, (48, 48), interpolation=cv2.INTER_LANCZOS4)
  gray_1d = np.mean(resized, axis=-1)
  gray = np.zeros_like(resized)
  gray[:,:,0] = gray_1d
  gray[:,:,1] = gray_1d
  gray[:,:,2] = gray_1d
  normalized = gray/255
  model_input = np.expand_dims(normalized,0)

  # Run the model
  scores_transfer = model.predict(model_input)
  #scores_lm_model = model_lm.predict(model_input.reshape())

  with st.spinner(text='predicting ...'):
      time.sleep(5)
      st.success('Prediction done')


  # Print results and plot score
  st.write(f"The predicted emotion with transfer learning is: {EMOTIONS[scores_transfer.argmax()]}")


  df = pd.DataFrame(scores_transfer.flatten(), columns = EMOTIONS)
  #df["Emotion"] = EMOTIONS
  #df["Scores_transfer"] = scores_transfer.flatten()


  st.area_chart(df)
  st.balloons()

  
