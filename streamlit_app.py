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

  st.title("Facial Emotion Detector")
  

  app_mode = st.sidebar.selectbox('You may choose other modes here: ', ['App demo', 'Technical specifications'])
  model = download_model()
  if app_mode == 'App demo':
    st.header("This app detects your emotions! upload a picture to try it out!")
    st.subheader("Neural network is trained to recognize the human facial emotions")
    st.write("it has overall 70% percent accuracy which is 5% better than average human, computer better than human in recognizing human emotions?!!!")
    f = st.file_uploader("Upload Image")

    if f is not None: 
      # Extract and display the image
      file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
      image = cv2.imdecode(file_bytes, 1)
      out = cv2.resize(image, (512, 512))
      st.image(out, channels="BGR")

      # Prepare the image
      resized = cv2.resize(image, (48, 48))
      
      img = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
      img = img/255
      model_input = img.reshape(1,48,48,1)

      # Run the model
      scores = model.predict(model_input)
      #scores_lm_model = model_lm.predict(model_input.reshape())

      with st.spinner(text='predicting ...'):
          time.sleep(5)
          st.success('Prediction done')

      #st.balloons()
      # Print results and plot score
      st.write(f"The predicted emotion: {EMOTIONS[scores.argmax()]}")
      st.warning("Here is overall results:")
      st.write(" ")
      col1, col2 , col3, col4, col5 = st.columns(5)
      
      
      col1.metric(EMOTIONS[0], str(scores[0][0]*100)+" %")
      col2.metric(EMOTIONS[1], str(scores[0][1]*100)+" %")
      col3.metric(EMOTIONS[2], str(scores[0][2]*100)+" %")
      col4.metric(EMOTIONS[3], str(scores[0][3]*100)+" %")
      col5.metric(EMOTIONS[4], str(scores[0][4]*100)+" %")
    
      st.balloons()
      
  elif app_mode == 'Neural Network details':
    st.subheader("Neural network test performance and network details")

    cm_img, msummary_img = download_images()

    st.image(cm_img,caption="Confusion Matrix details on test data")

    st.markdown("\n")
    st.markdown("\n")
    st.image(msummary_img, caption="Neural network architecture details")



@st.cache(allow_output_mutation=True)
def download_model():
  model_link = 'https://www.dropbox.com/s/90m9p48gkkfd54i/best_cnn_model_2.h5?dl=1'
  
  model_file = 'model.h5'

  wget.download(model_link,model_file)

  
  # accuracy = 70% ~ 5% better than human in TELLING EMOTIONS!!! (on average)
  loaded_model = tensorflow.keras.models.load_model('model.h5')
  return loaded_model


@st.cache()
def download_images():
  cm_link = 'https://www.dropbox.com/s/c3ufe0a1ukq5ilx/Screen%20Shot%202021-09-29%20at%203.45.16%20PM.png?dl=1'
  m_summary_link = 'https://www.dropbox.com/s/fmgzzg16dvi8tbb/Screen%20Shot%202021-09-29%20at%203.55.06%20PM.png?dl=1'

  cm = 'cm.png'
  msummary = 'summary.png'

  wget.download(cm_link,cm)
  wget.download(m_summary_link,msummary)

  return cm, msummary


if __name__ == "__main__":
  main()
