# from deepface import DeepFace
# import cv2
# import os
# import pandas as pd  # Ensure pandas is imported
# db_path='Data/RecognitionData'
# # Check for the existence of the authorized images directory
# if not os.path.exists(db_path):
#     print("Authorized images directory does not exist.")
#     exit()

# # Initialize webcam
# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open video capture.")
#     exit()

# # Load OpenCV's Haar Cascade for face detection
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# def verify_with_authorized(face_image_path):
#     try:
#         result = DeepFace.find(img_path=face_image_path, db_path=db_path, model_name="Facenet", enforce_detection=False)
#         print("result",result[0])
#         # Check if the result is as expected
#         if isinstance(result, list) and len(result) > 0 and all(isinstance(item, pd.DataFrame) for item in result):
#             df = result[0]
#             print("type",type(df))
#             # Calculate and print the average distance if the DataFrame is not empty
#             average_distance = df['distance'].mean()
#             print(f"Average distance: {average_distance}")
                
#             if average_distance < 0.35:
#                 print("Authorized user detected based on average distance.")
#                 return True
#             else:
#                 print("Unauthorized user detected based on average distance.")
#                 return False
            
#         else:
#             print("No matches found or result is not in expected format.")
#             return False
#     except Exception as e:
#         print(f"Verification failed: {e}")
#         return False
# try:
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture frame from camera. Exiting...")
#             break

#         # Convert frame to grayscale for face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         if faces is not None and len(faces) > 0:
#             for (x, y, w, h) in faces:
#                 # Draw rectangle around the face
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
#                 # Crop the detected face
#                 cropped_face = frame[y:y+h, x:x+w]

#                 # Save the cropped face to a temporary file to use in verification
#                 temp_face_path = "temp_face.jpg"
#                 cv2.imwrite(temp_face_path, cropped_face)

#                 # Verify if the cropped face is authorized
#                 if verify_with_authorized(temp_face_path):
#                     verification_text = "Authorized user detected."
#                     cv2.putText(frame, verification_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                 else:
#                     verification_text = "Unauthorized user detected."
#                     cv2.putText(frame, verification_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#                 # Clean up temporary file
#                 if os.path.exists(temp_face_path):
#                     os.remove(temp_face_path)
#         else:
#             cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#         cv2.imshow('Camera Output', frame)

#         # Break the loop with 'q'
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
# finally:
#     cap.release()
#     cv2.destroyAllWindows()
from deepface import DeepFace
import cv2
import os
import pandas as pd

db_path = 'Data/RecognitionData'

if not os.path.exists(db_path):
    print("Authorized images directory does not exist.")
    exit()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

def process_frame(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cropped_face = frame[y:y+h, x:x+w]
        temp_face_path = "temp_face.jpg"
        cv2.imwrite(temp_face_path, cropped_face)

        if verify_with_authorized(temp_face_path):
            cv2.putText(frame, "Authorized user detected.", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unauthorized user detected.", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        os.remove(temp_face_path)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Camera Output', frame)

def main_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        exit()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera. Exiting...")
                break

            process_frame(frame, face_cascade)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
