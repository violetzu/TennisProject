在主目錄下Build Dockerfile
```
cd ~/TennisProject
docker build -f docker/Dockerfile -t tennis:latest .
```
Run Container
```
docker run --gpus all -it --rm \
  -v ~/TennisProject:/workspace \
  -p 8000:8000 \
  tennis:latest /bin/bash
```