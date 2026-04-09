FROM runpod/base:0.4.0-cuda11.8.0

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    runpod>=1.6.0 \
    deepface==0.0.93 \
    opencv-python-headless==4.9.0.80 \
    numpy>=1.24 \
    tf-keras

COPY handler.py .

# Pre-download Facenet weights at build time (avoids cold-start download)
RUN python -c "
from deepface import DeepFace
import numpy as np
dummy = np.zeros((160,160,3), dtype='uint8')
DeepFace.represent(dummy, model_name='Facenet', enforce_detection=False)
print('Facenet weights downloaded OK')
"

CMD ["python", "-u", "handler.py"]
