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
st.markdown(footer, unsafe_allow_html=True)



st.title('Cat Breed Classifier')

upload_image = st.file_uploader("Upload your cat's image:", type=['jpg', 'jpeg', 'png'])
if upload_image is not None:
    predictions = predict(upload_image)

    st.write('## Meow!! Here are the top 3 predictions:')
    for prediction in predictions: 
        st.write(f"{prediction['label']} - {prediction['proba']:.2f}%")

    st.warning(
        """
        The classifier has been trained only on 12 common breeds (listed below), wrong prediction or weak confidence\
        in a prediction is maybe due the model not being trained on that specific breed.

        The breeds trained are: Abyssinian, Bengal, Birman, Bombay, British Shorthair,
       Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue,
       Siamese, and Sphynx.
        """
    )