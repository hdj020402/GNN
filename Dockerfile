FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app
COPY . .
RUN pip install -r envs/requirements.txt

# CMD ["python3", "main.py"]