# IsaNLP SRL Demo 

This directory contains docker compose file to start IsaNLP containers 
and REST service to communicate with them.

## Build

```
./build.sh
```


## Run

```
./run.sh
```

## Communicate

```
curl -X POST \
  http://localhost:3333/api/srl \
  -H 'Content-Type: application/json' \
  -d '{"text": "Мы поехали на дачу."}'
```
