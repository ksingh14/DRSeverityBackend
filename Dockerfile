FROM tiangolo/uwsgi-nginx-flask:latest

COPY './requirements.txt' .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
# RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# Setup container directories
RUN mkdir -p /app

# Copy local code to the container
COPY ./app /app

WORKDIR /app
EXPOSE 8080
# ENTRYPOINT ["python"]
# CMD ["app/main.py"]
# CMD ["gunicorn", "main:app", "--timeout=0", "--preload", "--reload",\
#      "--workers=4", "--threads=4", "--bind=0.0.0.0:8080"]

CMD ["uwsgi", "--http=0.0.0.0:8080", "--wsgi-file=main.py", \
     "--callable=app", "--processes=2", "--threads=2", \
     "--cheaper=0", "--listen=100", \
     "--lazy-apps"]