FROM python:3.10-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas numpy scikit-learn boto3 skl2onnx onnxruntime

COPY train.py /app/train.py

ENTRYPOINT ["python", "-u", "train.py"]