FROM pytorch/pytorch

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 mariadb -y

WORKDIR /service
ADD . /service
RUN pip install -r requirements.txt

EXPOSE 32332

CMD ["python", "service.py", "--port=32332", "--model=YVCR_big"]