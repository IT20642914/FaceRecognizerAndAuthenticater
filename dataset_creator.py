# import cv2
# import numpy as np
# import os
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from sklearn.model_selection import train_test_split
# import shutil

# # Face detection setup
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Initialize Keras Data Augmentation
# datagen = ImageDataGenerator(
#     rotation_range=10,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     brightness_range=(0.8, 1.2),
#     shear_range=0.1,
#     zoom_range=0.1,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )

# # Function to apply data augmentations using Keras
# def apply_augmentations_keras(image):
#     image = image.reshape((1,) + image.shape)  # Reshape to (1, height, width, channels)
#     it = datagen.flow(image, batch_size=1)
#     batch = next(it)
#     augmented_image = batch[0].astype('uint8')
#     return augmented_image

# # Get video file path and user information
# video_path = '20240312_104654.mp4'  # Update this path as necessary

# # Open the video file
# video_capture = cv2.VideoCapture(video_path)

# dataset_folder = 'Data/Dataset'
# if not os.path.exists(dataset_folder):
#     os.makedirs(dataset_folder)

# sample_faces = 0
# frame_skip_interval = 5  # To control speed

# while sample_faces < 600:
#     ret, frame = video_capture.read()

#     if not ret:
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#     if len(faces) > 0:
#         for (x, y, w, h) in faces:
#             sample_faces += 1
#             face_image = frame[y:y+h, x:x+w]
#             face_image = cv2.resize(face_image, (250, 250))
#             augmented_image = apply_augmentations_keras(face_image.copy())

#             face_save_path = os.path.join(dataset_folder, f'{sample_faces}_face.jpg')
#             cv2.imwrite(face_save_path, face_image)

#             augmented_save_path = os.path.join(dataset_folder, f'{sample_faces}_augmented.jpg')
#             cv2.imwrite(augmented_save_path, augmented_image)

# video_capture.release()
# cv2.destroyAllWindows()

# # Ensure images were saved before proceeding
# all_images = os.listdir(dataset_folder)
# if len(all_images) == 0:
#     print("No images were saved to the dataset. Please check the video file and face detection.")
# else:
#     # Proceed with splitting the dataset
#     train_images, validation_images = train_test_split(all_images, test_size=0.05, random_state=42)

#     # Create folders for recognition and verification data
#     recognition_data_folder = 'Data/RecognitionData'
#     verification_data_folder = 'Data/VerificationData'

#     if not os.path.exists(recognition_data_folder):
#         os.makedirs(recognition_data_folder)
#     if not os.path.exists(verification_data_folder):
#         os.makedirs(verification_data_folder)

#     # Move the files
#     for image in train_images:
#         shutil.move(os.path.join(dataset_folder, image), os.path.join(recognition_data_folder, image))

#     for image in validation_images:
#         shutil.move(os.path.join(dataset_folder, image), os.path.join(verification_data_folder, image))

#     print("Data has been successfully divided and moved.")
from flask import Flask, request, jsonify
import cv2
import numpy as np
import os
import requests
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil
import firebase_admin
from firebase_admin import credentials, db

# Initialize Flask app
app = Flask(__name__)

def initialize_firebase():
    cred = credentials.Certificate(r"private_key.json")
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://suranimala-8e0c4-default-rtdb.firebaseio.com/'})

initialize_firebase()

# Face detection setup
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize Keras Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    brightness_range=(0.8, 1.2),
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Function to download video from Firebase Storage URL
def download_video(url, output_path):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(output_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
    else:
        print("Failed to download video")
        return None
    return output_path

# Function to get the current active user and their verification video URL from Firebase Realtime Database
def get_active_user_video_url():
    try:
        ref = db.reference('Process')  # Update this path
        active_user_data = ref.get()
        if active_user_data:
            email = active_user_data['currentActiveUser']
            user_ref = db.reference('users')  # Update this path
            users = user_ref.get()
            for user_id, user_data in users.items():
                print(f"Checking user: {user_data}")  # Debugging line
                if user_data['email'] == email:
                    if 'verificationVideoURL' in user_data:
                        return user_data['verificationVideoURL']
                    else:
                        print(f"No verificationVideoURL for user: {user_data}")
                        return None
        return None
    except KeyError as e:
        print(f"KeyError: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Function to apply data augmentations using Keras
def apply_augmentations_keras(image):
    image = image.reshape((1,) + image.shape)  # Reshape to (1, height, width, channels)
    it = datagen.flow(image, batch_size=1)
    batch = next(it)
    augmented_image = batch[0].astype('uint8')
    return augmented_image

@app.route('/process-video', methods=['POST'])
def process_video():
    try:
        # Get the verification video URL
        video_url = get_active_user_video_url()
        if not video_url:
            return jsonify({"error": "Verification video URL not set, process stopped."}), 400

        video_path = download_video(video_url, 'downloaded_video.mp4')
        if not video_path:
            return jsonify({"error": "Failed to download the video."}), 500

        # Open the video file
        video_capture = cv2.VideoCapture(video_path)

        dataset_folder = 'Data/Dataset'
        if not os.path.exists(dataset_folder):
            os.makedirs(dataset_folder)

        sample_faces = 0
        frame_skip_interval = 5  # To control speed

        while sample_faces < 600:
            ret, frame = video_capture.read()

            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    sample_faces += 1
                    face_image = frame[y:y+h, x:x+w]
                    face_image = cv2.resize(face_image, (250, 250))
                    augmented_image = apply_augmentations_keras(face_image.copy())

                    face_save_path = os.path.join(dataset_folder, f'{sample_faces}_face.jpg')
                    cv2.imwrite(face_save_path, face_image)

                    augmented_save_path = os.path.join(dataset_folder, f'{sample_faces}_augmented.jpg')
                    cv2.imwrite(augmented_save_path, augmented_image)

        video_capture.release()
        cv2.destroyAllWindows()

        # Ensure images were saved before proceeding
        all_images = os.listdir(dataset_folder)
        if len(all_images) == 0:
            return jsonify({"error": "No images were saved to the dataset. Please check the video file and face detection."}), 500

        # Proceed with splitting the dataset
        train_images, validation_images = train_test_split(all_images, test_size=0.05, random_state=42)

        # Create folders for recognition and verification data
        recognition_data_folder = 'Data/RecognitionData'
        verification_data_folder = 'Data/VerificationData'

        if not os.path.exists(recognition_data_folder):
            os.makedirs(recognition_data_folder)
        if not os.path.exists(verification_data_folder):
            os.makedirs(verification_data_folder)

        # Move the files
        for image in train_images:
            shutil.move(os.path.join(dataset_folder, image), os.path.join(recognition_data_folder, image))

        for image in validation_images:
            shutil.move(os.path.join(dataset_folder, image), os.path.join(verification_data_folder, image))

        return jsonify({"message": "Data has been successfully divided and moved."})
    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "An error occurred during processing."}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6011)
