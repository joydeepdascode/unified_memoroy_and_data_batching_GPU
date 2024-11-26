# src/config.py
import os

# Manually set the CUDA path for the DLLs
cuda_path = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin"
ffmpeg_path = r"C:\Users\Joydeep Das\Downloads\ffmpeg-2024-11-25-git-04ce01df0b-essentials_build\ffmpeg-2024-11-25-git-04ce01df0b-essentials_build\bin"

# Add the CUDA path to the system PATH environment variable
os.environ['PATH'] = cuda_path + ";" + os.environ['PATH']

# Add the ffmpeg path to the system PATH environment variable
os.environ['PATH'] = ffmpeg_path + ";" + os.environ['PATH']
