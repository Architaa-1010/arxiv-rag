FROM python:3.11-slim

WORKDIR /app

# install dependencies
COPY requirements_deploy.txt .
RUN pip install --no-cache-dir -r requirements_deploy.txt

# copy app files
COPY app_deploy.py .
COPY chroma_deploy.zip .
COPY frontend/ frontend/

# expose port
EXPOSE 7860

# run streamlit on HF's required port
CMD ["streamlit", "run", "app_deploy.py", "--server.port=7860", "--server.address=0.0.0.0"]