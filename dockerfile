# Use Python 3.11 slim (Debian Trixie base - works with MediaPipe)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install updated OpenCV/MediaPipe dependencies (no obsolete packages)
RUN apt-get update && apt-get install -y \
    # GLX/OpenGL support (replaces libgl1-mesa-glx)
    libgl1 \
    libglx-mesa0 \
    # General deps for OpenCV video/GStreamer
    libglib2.0-0 \
    # Webcam/video capture (if needed for cv2.VideoCapture)
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy your script
COPY . .

# Run your face mesh script
CMD ["python", "index.py"]