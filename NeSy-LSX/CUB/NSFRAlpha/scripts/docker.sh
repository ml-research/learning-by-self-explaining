docker build -t nsfr .
docker run -v /home/ml-hshindo/Workspace/nsfr:/NSFR -itd --gpus all --shm-size 200G  nsfr