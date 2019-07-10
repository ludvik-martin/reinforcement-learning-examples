docker run --rm -it \
    -w /app
    -v `pwd`/..:/app \
    -v /tmp/gym-results:/tmp/gym-results \
    -p 8888:8888  \
    -u $(id -u):$(id -g) \
    mludvik/gym:2.0.latest
