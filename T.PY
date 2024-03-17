import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import pyaudio
import numpy as np
import queue
import threading
import time
import wave
import os
import whisper
import pyttsx3
import winsound

# Initialize Firebase Admin SDK with the service account key JSON file
cred = credentials.Certificate(r"C:\AMD\private_key.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://suranimala-8e0c4-default-rtdb.firebaseio.com/'
})

# Get a reference to the Realtime Database service
ref = db.reference('/voice_command')

# Function to send data to Firebase
def send_to_firebase(command):
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    data = {
        "command": command,
        "timestamp": current_time
    }
    ref.push(data)  # Pushing data directly to '/voice_command'


# Function to continuously read audio from the microphone and put it into the queue
def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
    
def record_audio(duration, sample_rate=44100, channels=1):
    print(f"Please speak. Recording for {duration} seconds...")
    frames = []
    audio = pyaudio.PyAudio()
    try:
        stream = audio.open(format=pyaudio.paInt16,
                            channels=channels,
                            rate=sample_rate,
                            input=True,
                            frames_per_buffer=1024,
                            stream_callback=audio_callback)
        stream.start_stream()
        start_time = time.time()
        while time.time() - start_time < duration:
            data = audio_queue.get()
            frames.append(data)
        return np.frombuffer(b''.join(frames), dtype=np.int16)
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()
speak("Face Authentication completed.")
speak("Drone moved to drone controlling mode. Follow the instructions for controlling the drone using voice commands. Each voice command directs the drone to move 1 meter in duration. Please ensure do not cover the face.")
speak("Drone control commands are: Forward, Backward, going-Left, going-Right, going-Up, going-Down, drone-Landing. ")
speak("you can give commands to me, After the beep sound.")
speak("If the landing position is OK or not.")
winsound.Beep(1000, 1000)
time.sleep(0.2)
def transcribe_audio(audio_data):
    # Save audio data to a file
    current_directory = os.getcwd()
    audio_file = os.path.join(current_directory, "input.wav")
    save_audio_to_wav(audio_data, audio_file)
   
    speak("please wait util given command is executed.")

    # Transcribe audio
    model = whisper.load_model("base")
    result = model.transcribe(audio_file)
    transcription_text = result["text"]
    print("Transcription:", transcription_text)

    # Check if transcription contains relevant keywords and print corresponding commands
    if "ok" in transcription_text.lower() or "okay" in transcription_text.lower():
        print("stop")
        send_to_firebase("stop")  # Sending "stop" command to Firebase
        speak("ok, changed drone mode as the landing.auto landing process starting.")
        return True

    elif "no" in transcription_text.lower() or "no." in transcription_text.lower() or "oh." in transcription_text.lower():
        print("no")
        # speak("NO, after beep sound you can give me commands for me.")
        # winsound.Beep(1000, 1000)

    elif "left" in transcription_text.lower() or "lift" in transcription_text.lower() or "lift." in transcription_text.lower():
        print("Drone is moving to the left")
        send_to_firebase("going_left")  # Sending "going_left" command to Firebase
        speak("Drone is moving left")

    elif "right" in transcription_text.lower() or "right." in transcription_text.lower():
        print("Drone is moving to the right")
        send_to_firebase("going_right")  # Sending "going_right" command to Firebase
        speak("Drone is moving right")

    elif "up" in transcription_text.lower():
        print("Drone is moving up")
        send_to_firebase("going_up")  # Sending "going_up" command to Firebase
        speak("Drone is moving up")

    elif "down" in transcription_text.lower():
        print("Drone is moving down")
        send_to_firebase("going_down")  # Sending "going_down" command to Firebase
        speak("Drone is moving down")


    
         # Indicate to stop

    # Print the transcription text
    print("Transcription:", transcription_text)

    return False  # Continue running

def save_audio_to_wav(audio_data, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Assuming mono audio
        wf.setsampwidth(2)  # 16-bit samples
        wf.setframerate(44100)
        wf.writeframes(audio_data.tobytes())
    print(f"Audio saved to {filename}")

if _name_ == "_main_":
    audio_queue = queue.Queue()

    try:
        while True:
            recorded_audio = record_audio(2)
            if recorded_audio is None:
                continue

            if transcribe_audio(recorded_audio):
                print("Stopping program.")
                break  # Exit the loop if transcription indicates to stop
            
            speak("After the beep sound, you can give me commands.If the landing position is okay, do nothing; otherwise, give me a command")
            winsound.Beep(1000, 1000)
            print("Please speak.")
            # time.sleep(0.5)

    except KeyboardInterrupt:
        print("Program terminated by user.")
    except Exception as e:
        print("An unexpected error occurred:", e)