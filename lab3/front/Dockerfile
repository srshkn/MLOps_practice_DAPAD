FROM python:3.12-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir streamlit requests
EXPOSE 8501
CMD ["streamlit", "run", "front.py", "--server.address=0.0.0.0", "--server.port=8501"]