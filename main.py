import time
import cv2
import os
import pandas as pd
import pyttsx3
import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, request, jsonify
import threading
import requests
import numpy as np

app = Flask(__name__)

# Configuration paths
DB_PATH = 'Data/RecognitionData'
TARGET_IMAGE_PATH = "Data/InputImage/authorized_user.jpg"
VERIFICATION_FOLDER_PATH = "Data/VerificationData"

# Initialize global variables for thread management
face_recognition_thread = None
face_recognition_running = False

# Firebase Initialization
def initialize_firebase():
    try:
        cred = credentials.Certificate("private_key.json")
        firebase_admin.initialize_app(cred, {'databaseURL': 'https://your-database-url.firebaseio.com/'})
    except Exception as e:
        print(f"Failed to initialize Firebase: {str(e)}")
        exit(1)

initialize_firebase()
ref = db.reference('/authorized_users')

def send_to_firebase(command):
    try:
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        data = {"status": command, "timestamp": current_time}
        ref.push(data)
    except Exception as e:
        print(f"Failed to send data to Firebase: {str(e)}")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def speak(text):
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print(f"Error using text-to-speech: {str(e)}")

def verify_face(target_image_path, verification_folder_path):
    try:
        verification_images = [os.path.join(verification_folder_path, f) for f in os.listdir(verification_folder_path) if os.path.isfile(os.path.join(verification_folder_path, f))]
        fixed_file_paths = [path.replace('\\', '/') for path in verification_images]
        distances = []

        for img_path in fixed_file_paths:
            result = DeepFace.verify(img1_path=img_path, img2_path=target_image_path, model_name='Facenet', enforce_detection=False)
            distances.append(result['distance'])

        if distances:
            average_distance = sum(distances) / len(distances)
            if average_distance < 0.5:
                return True
        return False
    except Exception as e:
        print(f"Error verifying target against folder: {str(e)}")
        return False

def fetch_and_decode_mjpeg(stream_url):
    try:
        stream = requests.get(stream_url, stream=True)
        bytes = b''
        for chunk in stream.iter_content(chunk_size=1024):
            bytes += chunk
            a = bytes.find(b'\xff\xd8')
            b = bytes.find(b'\xff\xd9')
            if a != -1 and b != -1:
                jpg = bytes[a:b+2]
                bytes = bytes[b+2:]
                image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is not None:
                    yield image
    except Exception as e:
        print(f"Error fetching and decoding MJPEG: {str(e)}")

def main_loop(stream_url):
    try:
        speak("Initiating face recognition protocol. Please look into the camera.")
        face_detected_time = None
        for frame in fetch_and_decode_mjpeg(stream_url):
            process_frame(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()
    except Exception as e:
        print(f"Error in main loop: {str(e)}")
        cv2.destroyAllWindows()

def start_face_recognition(stream_url):
    thread = threading.Thread(target=main_loop, args=(stream_url,))
    thread.start()

@app.route('/start_recognition', methods=['GET'])
def trigger_face_recognition():
    start_face_recognition('http://192.168.43.189:8080/?action=stream')
    return jsonify({"message": "Face recognition process started."})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6006)
