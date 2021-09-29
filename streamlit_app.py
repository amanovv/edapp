import streamlit as st
import tensorflow
import tensorflow.keras
import pandas as pd 
import cv2
import time
import wget
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
    

    # Prepare the image
    resized = cv2.resize(image, (48, 48))
    st.image(resized, channels="BGR")
    img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    img = img/255
    model_input = img.reshape(1,48,48,1)

    # Run the model
    scores = model.predict(model_input)
    #scores_lm_model = model_lm.predict(model_input.reshape())

    with st.spinner(text='predicting ...'):
        time.sleep(5)
        st.success('Prediction done')

    st.balloons()
    # Print results and plot score
    st.write(f"The predicted emotion: {EMOTIONS[scores.argmax()]}")

    col1, col2 , col3, col4, col5 = st.columns(5)

    col1.metric(EMOTIONS[0], str(scores[0]*100)+" %")
    col2.metric(EMOTIONS[1], str(scores[1]*100)+" %")
    col3.metric(EMOTIONS[2], str(scores[2]*100)+" %")
    col4.metric(EMOTIONS[3], str(scores[3]*100)+" %")
    col5.metric(EMOTIONS[4], str(scores[4]*100)+" %")
    
    

    #st.write(scores_transfer.flatten())
    #df = pd.DataFrame(scores_transfer.flatten(), columns = EMOTIONS)
    #df["Emotion"] = EMOTIONS
    #df["Scores_transfer"] = scores_transfer.flatten()


    #st.area_chart(df)
    st.balloons()

@st.cache(allow_output_mutation=True)
def download_model():
  model_link = 'https://www.dropbox.com/s/90m9p48gkkfd54i/best_cnn_model_2.h5?dl=1'
  
  model_file = 'model.h5'

  wget.download(model_link,model_file)

  
  # accuracy = 70% ~ 5% better than human in TELLING EMOTIONS!!! (on average)
  loaded_model = tensorflow.keras.models.load_model('model.h5')
  return loaded_model


if __name__ == "__main__":
  main()
