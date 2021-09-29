import streamlit as st
import tensorflow
import pandas as pd 
import cv2
import time
import wget
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow.keras.optimizers as optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, MaxPooling2D, Dropout, Flatten, Reshape, Dense, Conv2D
import numpy as np

def main():
  EMOTIONS = ['ANGRY', 'HAPPY', 'SAD', 'SURPRISE', 'NEUTRAL']

  st.title("Emotion Detector")
  st.header("This app detects your emotions! upload a picture to try it out!")
  model = download_model()
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

    st.balloons()
    # Print results and plot score
    st.write(f"The predicted emotion with transfer learning is: {EMOTIONS[scores_transfer.argmax()]}")

    
    df = pd.DataFrame(scores_transfer.flatten(), columns = EMOTIONS)
    #df["Emotion"] = EMOTIONS
    #df["Scores_transfer"] = scores_transfer.flatten()


    st.area_chart(df)
    st.balloons()

st.cache()
def download_model():
  model_link = 'https://www.dropbox.com/s/bdyohk8pbmxummc/best_mlp_model.h5?dl=1'
  #while finished == 0:
  #  st.spinner()
  #  os.system(f"wget https://www.dropbox.com/s/072b5vf4b33bu1l/emotion_detection_model_for_streamlit.h5")
  #  finished = 1
  #filename = wget.download(url)
  model_file = "eda_streamlit.h5"
  wget.download(model_link,model_file)

  #filename = tf.keras.utils.get_file("emotion_detection_model_for_streamlit.h5", url)
  
  # accuracy = 68.25% ~ 3.25% better than human in TELLING EMOTIONS!!! (on average)
  model = tensorflow.keras.models.load_model("eda_streamlit.h5")
  return model


if __name__ == "__main__":
  main()
