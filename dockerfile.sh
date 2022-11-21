
export $(xargs < ./.env)

docker build -t code-scout-python \
    --build-arg AIMEDIC_GROUPER_VERSION=$AIMEDIC_GROUPER_VERSION \
    --build-arg AWS_REGION=$AWS_REGION \
    --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY . \
    && docker run  -p 8888:8888 -v `pwd`:/home/jovyan/work code-scout-python
