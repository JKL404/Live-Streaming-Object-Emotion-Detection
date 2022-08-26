import cv2
from detection import detect
import streamlit as st


@st.cache(allow_output_mutation=True)
def get_cap():
    return cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    camera = get_cap()
    frameST = st.empty()
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = detect.detect_object(camera)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
    frameST.image(frame, channels="BGR")
