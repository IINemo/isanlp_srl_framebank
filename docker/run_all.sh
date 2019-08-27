#!/usr/bin/env bash

echo "Cleaning up previous containers"
docker rm -f morphology syntaxnet srl demo || true

echo "Starting Morphology Server" &&
docker run -d --name morphology -p 3333:3333 inemo/isanlp &&
echo "Morphology started" &&
echo "Starting Syntaxnet" &&
docker run --shm-size=1024m -d --name syntaxnet -p 3344:9999 inemo/syntaxnet_eng server 0.0.0.0 9999 &&
echo "Syntaxnet started" &&
echo "Starting SRL" &&
docker run -d -p 3355:3333 --name srl tchewik/isanlp_srl_framebank &&
echo "SRL Started" &&
echo "Starting web-ui" &&
docker run -d -p 3366:80 --name demo -e HOSTNAME=localhost -e IP_ADDRESS=localhost -e MORPH_PORT=3333 -e SYNTAX_PORT=3344 -e SEM_PORT=3355 inemo/isanlp_srl_framebank_demo
