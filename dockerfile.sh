echo "--- Reading environment variables ---"
# shellcheck disable=SC2046
export $(xargs < ./.env)

echo "--- Building image ---"
# --force-rm: Always remove intermediate containers
# --pull: Always attempt to pull a newer version of the image
docker build -t code-scout-python \
    --force-rm \
    --pull \
    --progress=plain \
    --build-arg AWS_REGION=$AWS_REGION \
    --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
    --build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
    .

echo "--- Starting container ---"
CONTAINER_ID=$(docker run -it --rm --detach code-scout-python:latest)

echo "--- Copying aimedic-grouper JAR to host ---"
docker cp "$CONTAINER_ID":/opt/project/resources/jars/aimedic-grouper-assembly.jar "$(pwd)"/resources/jars/

echo "--- done ---"
