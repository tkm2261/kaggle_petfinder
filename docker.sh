docker run -v /mnt/takamisato/working:/working -p 7788:7788 --gpus all --shm-size=8gb --rm -it --shm-size 8gb gcr.io/kaggle-gpu-images/python jupyter lab --port=7788 --allow-root --ip=0.0.0.0
