# import cv2
# import numpy as np
# import os

# # Face detection setup
# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Function to apply data augmentations
# def apply_augmentations(image):
#     # Flipping
#     image = cv2.flip(image, 1) 

#     # Brightness and contrast variations
#     alpha = np.random.uniform(0.8, 1.2) 
#     beta = np.random.randint(-30, 30) 
#     image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta) 

#     # Rotations 
#     angle = np.random.randint(-10, 10) 
#     rows, cols = image.shape[:2]
#     rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
#     image = cv2.warpAffine(image, rotation_matrix, (cols, rows)) 

#     return image

# # Get video file path and user information
# video_path = '20240302_234719.mp4' 
# name = input('Enter user name: ')
# id = input('Enter user id: ')


# # Open the video file
# video_capture = cv2.VideoCapture(video_path)

# sample_faces = 0
# frame_skip_interval = 5  

# while sample_faces < 200: 
#     ret, frame = video_capture.read()

#     if not ret: 
#         print("End of video reached and still not enough images. Consider a different video.")
#         break

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5) 

#     if len(faces) > 0:
#         for (x, y, w, h) in faces:
#             sample_faces += 1
#             face_image = gray[y:y+h, x:x+w]
#             augmented_image = apply_augmentations(face_image.copy())
            
#              # Save original face image
#             face_save_path = os.path.join('dataset', f'{id}.{sample_faces}.{name}_face.jpg')
#             cv2.imwrite(face_save_path, face_image)
            
#             # Save augmented image
#             augmented_save_path = os.path.join('dataset', f'{id}.{sample_faces}.{name}_augmented.jpg')
#             cv2.imwrite(augmented_save_path, augmented_image)
            
#             # Optional Visualization 
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) 
#             cv2.imshow("Face Extraction", frame[y:y+h*2, x:x+w*2]) 

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break  # Exit option during visualization

# print("Dataset with 200 faces has been successfully created") 
# video_capture.release()
# cv2.destroyAllWindows()
import cv2
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
    # Ensure the image has 3 channels for color images
    image = image.reshape((1,) + image.shape)  # Reshape to (1, height, width, channels)
    it = datagen.flow(image, batch_size=1)
    batch = next(it)
    # No need to convert to uint8 here, as we're dealing with color images directly
    augmented_image = batch[0].astype('uint8')  # Keep the color channels
    return augmented_image

# Get video file path and user information
video_path = '20240312_104654.mp4'  # Update this path as necessary
name = 'avishka'
id = '1'

# Open the video file
video_capture = cv2.VideoCapture(video_path)

sample_faces = 0
frame_skip_interval = 5  # Skip every 5 frames to speed up the process

while sample_faces < 450:
    ret, frame = video_capture.read()

    if not ret:
        print("End of video reached and still not enough images. Consider a different video.")
        break

    # Use the original frame for face detection to avoid grayscale conversion
     # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # faces = face_cascade.detectMultiScale(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            sample_faces += 1
            face_image = frame[y:y+h, x:x+w]

            # Resize the face image to 250x250 before augmentation
            face_image = cv2.resize(face_image, (250, 250))

            augmented_image = apply_augmentations_keras(face_image.copy())
            
            # Save original and augmented face images in color
            face_save_path = os.path.join('data/authorized', f'{id}.{sample_faces}.{name}_face.jpg')
            cv2.imwrite(face_save_path, face_image)
            
            augmented_save_path = os.path.join('data/authorized', f'{id}.{sample_faces}.{name}_augmented.jpg')
            cv2.imwrite(augmented_save_path, augmented_image)
            
            # Optional visualization
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("Face Extraction", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break  # Exit option during visualization

print("Dataset with 300 faces has been successfully created.")
video_capture.release()
cv2.destroyAllWindows()
