import streamlit as st
from predictor import predict

footer = """
<style>

footer: {
    visibility: visible;
}
footer:after{
    content: ' by Ajay DS';
}
</style>
"""



st.title('Cat Breed Classifier')

upload_image = st.file_uploader('Upload cat image', type=['jpg', 'jpeg', 'png'])
st.markdown(footer, unsafe_allow_html=True)
if upload_image is not None:
    predictions = predict(upload_image)

    st.write('## Meow!! Here are the top 3 predictions:')
    for prediction in predictions: 
        st.write(f"{prediction['label']}: {prediction['proba']:.2f}%")