import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil

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

# Function to apply data augmentations using Keras
def apply_augmentations_keras(image):
    image = image.reshape((1,) + image.shape)  # Reshape to (1, height, width, channels)
    it = datagen.flow(image, batch_size=1)
    batch = next(it)
    augmented_image = batch[0].astype('uint8')
    return augmented_image

# Get video file path and user information
video_path = '20240312_104654.mp4'  # Update this path as necessary

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
    print("No images were saved to the dataset. Please check the video file and face detection.")
else:
    # Proceed with splitting the dataset
    train_images, validation_images = train_test_split(all_images, test_size=0.1, random_state=42)

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

    print("Data has been successfully divided and moved.")