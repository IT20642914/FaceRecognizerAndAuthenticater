from deepface import DeepFace
import os

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
    else:
        print("No distances recorded due to errors in processing.")
