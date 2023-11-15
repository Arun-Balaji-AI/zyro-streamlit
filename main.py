import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

#Removing Streamlit Hamburger and Footer
st.markdown(
    """<style>
    .eyeqlp51.st-emotion-cache-fblp2m.ex0cdmw0{
    visibility : hidden;
    }
    </style>
    """ , unsafe_allow_html=True
)

#Loading the image classifier
model = tf.keras.models.load_model("zyro_image_classification-01.h5")
st.title("Image Classifier")
st.write("An Image Classifier made for ZYRO hackathon")
st.header("Predictable Classes")
classes = ["Airplane" , "Automobile" , "Bird" , "Cat" , "Deer" , "Dog" , "Frog" , "Horse" , "Ship" , "Truck"]

for i , cls in enumerate(classes , start = 1):
    st.markdown(f"{i}. {cls}")

#Predict the images
def predict(image):
    img = tf.keras.preprocessing.image.load_img(image,target_size = (96,96))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = img / 255
    img = tf.image.resize(img , (96,96))
    img = np.array(img).reshape(1,96,96,3)
    #print(img.shape)
    prediction = model.predict(img)
    cls = np.argmax(prediction)

    return classes[cls]

image = st.file_uploader("Upload Image" , type = ['jpg' , 'jpeg' , 'png'])

if image is not None:
    if st.button("Predict"):
        st.image(image , width = 700)
        res = predict(image)
        st.subheader("The given image is a {}".format(res))


