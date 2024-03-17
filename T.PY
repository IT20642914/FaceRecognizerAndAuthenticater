import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO

stream_url = 'http://192.168.43.189:8080/?action=stream'

# Start a stream request session
session = requests.Session()
response = session.get(stream_url, stream=True)

# Initialize buffer and content type flag
buffer = b''
content_type = 'image/jpeg'

try:
    for chunk in response.iter_content(chunk_size=1024):
        buffer += chunk
        # Check if a valid JPEG header is present
        a = buffer.find(b'\xff\xd8')
        b = buffer.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = buffer[a:b+2]
            buffer = buffer[b+2:]
            frame = np.array(Image.open(BytesIO(jpg)))
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Streamed Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    cv2.destroyAllWindows()
