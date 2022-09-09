import streamlit as st
import cv2
import numpy as np
from streamlit_lottie import st_lottie
import requests

st.write("""
# Streamlit App with OpenCV
Face Detection.
""")

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie = "https://assets7.lottiefiles.com/datafiles/2skOBPHl4c8xFfU/data.json"

camera_container = st.empty()
picture = camera_container.camera_input("Take a picture")
anim = load_lottieurl(lottie)

detect_faces = st.checkbox('Faces', value=True)
face_model = st.selectbox('Face Model', 
                        ('haarcascade_frontalface_default.xml', 'haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_alt2.xml', 'haarcascade_frontalface_alt_tree.xml', 'haarcascade_profileface.xml'))
    
detect_eyes = st.checkbox('Eyes', value=True)
eye_model = st.selectbox('Eyes Model',
                    ('haarcascade_eye.xml', 'haarcascade_eye_tree_eyeglasses.xml', 'haarcascade_lefteye_2splits.xml', 'haarcascade_righteye_2splits.xml'))


if picture:
    camera_container.empty()
    st_lottie(anim, width="100px", key="faceid")
    with st.spinner('Wait for it...'):
        bytes_data = picture.getvalue()
        img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        # Load the cascade

        if face_model and eye_model:
            model = {
                        'Faces' : (face_model, detect_faces),
                        'Eyes' : (eye_model, detect_eyes)
                    }
            cascades = [cv2.CascadeClassifier("haarcascades/"+m) for m, s in model.values() if s]

            # Convert into grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces
            boxes = [c.detectMultiScale(gray, 1.1, 4) for c in cascades]
            sample = img
            
            for res in boxes:
                for (x, y, w, h) in res:
                    cv2.rectangle(sample, (x, y), (x+w, y+h), (255, 0, 0), 2)

            st.image(sample)
            
            cant = sum([1 for _, s in model.values() if s])

            for m, col in enumerate(st.columns(cant)):
                if list(model.values())[m][1]:
                    col.metric(list(model.keys())[m], len(boxes[m]))

            st.success('Done!')
            st.balloons()
