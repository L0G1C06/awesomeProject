FROM python:3.11-slim
WORKDIR /app
RUN pip install gradio==4.37.0 httpx==0.27.0
COPY app.py .
EXPOSE 7860
CMD ["python", "app.py"]