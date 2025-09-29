
build

```
docker build -t openmmlab-2.1.2-cu121 . 
```

run

```
docker run --gpus all -it --name tennisproject \
  -v ~/TennisProject:/workspace \
  openmmlab-2.1.2-cu121 bash
```

start

`docker start -ai tennisproject`
