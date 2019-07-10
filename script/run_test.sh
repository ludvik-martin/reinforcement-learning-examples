 docker run -it --rm \
    -v `pwd`/..:/opt/project \
    -v /tmp/logdir:/tmp/logdir \
    -w /opt/project \
    -u $(id -u):$(id -g) \
    mludvik/gym:2.0 python3 test/DeepQLearningTest.py
