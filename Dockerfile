#FROM 172.17.1.119:5555/litemind/acai_service:0.1
FROM python:3.6

RUN rm -rf  /etc/localtime && ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

COPY . /csv2graph

WORKDIR /csv2graph

CMD "sh" "-c" "echo nameserver 8.8.8.8 > /etc/resolv.conf"
RUN ["pip", "install", "-r", "requirements.txt"]
#RUN echo `ls /core`

ENTRYPOINT  ["python", "-m", "production_app"]