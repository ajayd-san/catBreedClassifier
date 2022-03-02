from distutils.command.upload import upload
import streamlit as st
from predictor import predict

st.title('Cat Breed Classifier')

upload_image = st.file_uploader('Upload cat image', type=['jpg', 'jpeg', 'png'])

if upload_image is not None:
    predictions = predict(upload_image)

    st.write('## Meow!! Here are the predictions:')
    for prediction in predictions: 
        st.write(f"{prediction['label']}: {prediction['proba']:.2f}%")