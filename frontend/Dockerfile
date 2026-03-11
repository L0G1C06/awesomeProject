FROM python:3.11-slim
WORKDIR /app
RUN pip install gradio==4.37.2 httpx==0.27.0 huggingface_hub==0.23.5
COPY app.py .
EXPOSE 7860
CMD ["python", "app.py"]
