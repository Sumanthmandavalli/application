import streamlit as st
from ultralytics import YOLO
import cv2
import smtplib
from email.message import EmailMessage
import vonage
import os
from pathlib import Path
import shutil
import tempfile


WORKING_DIR = Path(os.getcwd())

model = YOLO("best.pt")
def email_alert(subject, body, to):
    msg = EmailMessage()
    msg.set_content(body)
    msg['subject'] = subject
    msg['to'] = to

    user = "sumanth.m@infyzterminals.com"
    msg['from'] = user
    password = "hkqr nsaz gqdh fjcj"

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(user, password)
        server.send_message(msg)
        server.quit()
        st.success("Email sent successfully!")
    except Exception as e:
        st.error(f"Failed to send email: {e}")

def send_sms_alert(body, phone_number):
    client = vonage.Client(key="ae47b9f2", secret="Bt4D94EDjxgX2mso")
    sms = vonage.Sms(client)

    try:
        responseData = sms.send_message(
            {
                "from": "yard",
                "to": phone_number,
                "text": body
            }
        )

        if responseData["messages"][0]["status"] == "0":
            st.success("SMS sent successfully.")
        else:
            st.error(f"SMS failed with error: {responseData['messages'][0]['error-text']}")
    except Exception as e:
        st.error(f"Failed to send SMS: {e}")

def detect_and_alert(video_path):
    cap = cv2.VideoCapture(video_path)
    fall_detected = False

    col1, col2 = st.columns(2) 
    with col1:
        stframe = st.empty()
    with col2:
        alert_placeholder = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        no_ppe_detected = False

        for result in results:
            for box in result.boxes:
                label = model.names[int(box.cls)]
                if label == "NO-Mask" or label == "NO-Hardhat" or label == "NO-Safety-Vest":
                    no_ppe_detected = True

        if no_ppe_detected:
            alert_placeholder.error("PPE Alert: A person without PPE has been detected.")
            email_alert("PPE Alert", "A person without PPE has been detected.", "sumanthmandavalli608@gmail.com")
            phone_number = "918367531279"
            send_sms_alert("PPE Alert: A person without PPE has been detected.", phone_number)
        else:
            alert_placeholder.info("Monitoring...")

        annotated_frame = results[0].plot()
        stframe.image(annotated_frame, channels="BGR")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    st.title("PPE Detection and Alert System")

    video_source_option = st.selectbox("Select video source", ["Webcam", "Video File"])

    if video_source_option == "Webcam":
        video_source = 0
        if st.button("Run"):
            detect_and_alert(video_source)
    else:
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "mov", "avi", "jpg"])

        if uploaded_file is not None:
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())
                video_source = tmp_file.name
            if st.button("Run Detection"):
                detect_and_alert(video_source)
                os.remove(video_source)
        else:
            st.warning("Please upload a video file.")

    if video_source_option == "Webcam" or (uploaded_file is not None and st.button("Run Detection")):
        detect_and_alert(video_source)

if __name__ == '__main__':
    main()