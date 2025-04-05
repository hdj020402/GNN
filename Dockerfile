FROM python:3.13.2
WORKDIR /workspace/gnn
COPY envs/requirements-gpu.txt envs/
RUN pip install -r envs/requirements-gpu.txt
COPY . .
CMD ["python", "main.py"]
