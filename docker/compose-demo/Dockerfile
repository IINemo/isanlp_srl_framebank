FROM alpine:3.6
RUN apk add --no-cache git gcc g++ py3-pip python3-dev && \
	pip3 install --no-cache-dir Flask grpcio \
		git+https://github.com/IINemo/isanlp \
		git+https://github.com/IINemo/isanlp_srl_framebank
COPY main.py /app/
WORKDIR /app
EXPOSE 3333
RUN pip3 install protobuf
CMD ["python3", "main.py"]
