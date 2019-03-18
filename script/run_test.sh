 docker run -it --rm \
    -v /home/martin/work/github/reinforcement-learning-examples:/opt/project \
    -v /tmp/logdir:/tmp/logdir \
    mludvik/gym:2.0 /bin/bash \
    -c "cd /opt/project; python3 test/DeepQLearningTest.py"
