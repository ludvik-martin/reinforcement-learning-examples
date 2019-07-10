 docker run -it --rm \
    -v `pwd`/..:/opt/project \
    -v /tmp/logdir:/tmp/logdir \
    -w /opt/project \
    mludvik/gym:2.0 python3 test/ReinforceHyperoptTest.py
