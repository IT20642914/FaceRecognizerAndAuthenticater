import time
from deepface import DeepFace
import cv2
import os
import pandas as pd
import pyttsx3
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from flask import Flask, request, jsonify
import threading


app = Flask(__name__)

db_path = 'Data/RecognitionData'
target_image_path = "Data/InputImage/authorized_user.jpg"
verification_folder_path = "Data/VerificationData"
# Initialize global variables for thread management
face_recognition_thread = None
face_recognition_running = False

if not os.path.exists(db_path):
    print("Authorized images directory does not exist.")
    exit()

def initialize_firebase():
    # Initialize Firebase Admin SDK with the service account key JSON file
    cred = credentials.Certificate(r"private_key.json")
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://suranimala-8e0c4-default-rtdb.firebaseio.com/'})
    # Get a reference to the Realtime Database service
initialize_firebase()
ref = db.reference('/authorized_users')

# Function to send data to Firebase
def send_to_firebase(command):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    data = {
        "status": command,
        "timestamp": current_time
    }
    ref.push(data)  


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    
def verify_target_against_folder(target_image_path, verification_folder_path):
    verification_images = [os.path.join(verification_folder_path, f) for f in os.listdir(verification_folder_path) if os.path.isfile(os.path.join(verification_folder_path, f))]
    fixed_file_paths = [path.replace('\\', '/') for path in verification_images]
    distances = []

    for img_path in fixed_file_paths:
        try:
            result = DeepFace.verify(img1_path=img_path, img2_path=target_image_path, model_name='Facenet', enforce_detection=False)
            print(f"Verification with {os.path.basename(img_path)}: {'Similar' if result['verified'] else 'Not Similar'} - Confidence: {result['distance']}")
            distances.append(result['distance'])
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    if distances:
        average_distance = sum(distances) / len(distances)
        print(f"Average Distance: {average_distance}")
        if(average_distance < 0.4):
            print("Authorized user detected based on average distance.")
            return True  
        else:
            return False
    else:
        print("No distances recorded due to errors in processing.")
        return False

def verify_with_authorized(face_image_path):
    try:
        result = DeepFace.find(img_path=face_image_path, db_path=db_path, model_name="Facenet", enforce_detection=False)
        print("result", result[0])
        if isinstance(result, list) and len(result) > 0 and all(isinstance(item, pd.DataFrame) for item in result):
            df = result[0]
            print("type", type(df))
            average_distance = df['distance'].mean()
            print(f"Average distance: {average_distance}")
            if average_distance < 0.35:
                print("Authorized user detected based on average distance.")
                return True
            else:
                print("Unauthorized user detected based on average distance.")
                return False
        else:
            print("No matches found or result is not in expected format.")
            return False
    except Exception as e:
        print(f"Verification failed: {e}")
        return False
def process_frame(frame, face_cascade, face_detected_time=None):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    current_time = time.time()
    if faces is not None and len(faces) > 0:
        if face_detected_time is None:
            face_detected_time = current_time

        for (x, y, w, h) in faces:
            if current_time - face_detected_time >= 2:  # If face has been recognized continuously for at least 2 seconds
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cropped_face = frame[y:y+h, x:x+w]
                temp_face_path = "Data/InputImage/authorized_user.jpg"
                cv2.imwrite(temp_face_path, cropped_face)

                if verify_with_authorized(temp_face_path):
                    speak("Face successfully recognized. Please wait while verification is in progress.")
                    cv2.putText(frame, "Authorized user detected.", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    time.sleep(2)
                    verify_response = verify_target_against_folder(target_image_path, verification_folder_path)
                    if(verify_response):
                        speak("Face verification successful. Access granted.")
                        send_to_firebase('Verified')
                        print("Authorized user detected based on average distance.")
                        return True, None
                    else:
                        speak("Face verification failed. Restarting face recognition. Please ensure clear visibility towards the camera.")
                        send_to_firebase('UnVerified')
                        print("Unauthorized user detected based on average distance.")
                        return False, None
                else:
                    cv2.putText(frame, "Unauthorized user detected.", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                os.remove(temp_face_path)
            else:
                cv2.putText(frame, "Detecting...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    else:
        face_detected_time = None  # Reset detection timer if no face is detected

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Camera Output', frame)
    return False, face_detected_time
stream_url = 'http://192.168.43.189:8080/?action=stream'
def main_loop():
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()

    speak("Initiating face recognition protocol. Kindly direct your gaze towards the drone's camera for verification. Please remain still during the process.")

    face_detected_time = None
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera. Exiting...")
                break

            IsSTopTrue, face_detected_time = process_frame(frame, face_cascade, face_detected_time)
            if(IsSTopTrue):
                break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def start_face_recognition():
    main_loop()
    
@app.route('/start_recognition', methods=['GET'])
def trigger_face_recognition():
    # Using a thread to run the face recognition process to avoid blocking Flask's main thread
    thread = threading.Thread(target=start_face_recognition)
    thread.start()
    return jsonify({"message": "Face recognition process started."})

if __name__ == "__main__":
     app.run(debug=True, port=5000) # Run the Flask app on port 5000