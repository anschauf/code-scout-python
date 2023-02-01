echo "--- Reading environment variables ---"
# shellcheck disable=SC2046
export $(xargs < ./.env)

echo "--- Building image ---"
# From the official documentation:
# --force-rm: `Always remove intermediate containers` => do not leave unneeded containers behind
# --no-cache: `Do not use cache when building the image` => Ignore images with the same name built in a different way
# --pull: `Always attempt to pull a newer version of the image` => Do not use cached base images
# With these settings, the build will be slower, because it skips the cache, but it is 100% reproducible because it
#   always builds it from scratch, ignoring similar images built via a different set of commands.
# `--progress=plain` logs every step in a more traditional way, instead of using a better-looking logger, which hides
#   some parts of the log, when a step is completed.
docker build -t code-scout-python \
    --force-rm \
    --no-cache \
    --pull \
    --progress=plain \
    --build-arg AWS_REGION="$AWS_REGION" \
    --build-arg AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
    --build-arg AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
    .

echo "--- Starting container ---"
CONTAINER_ID=$(docker run -it --rm --detach code-scout-python:latest)

echo "--- Copying aimedic-grouper JAR to host ---"
docker cp "$CONTAINER_ID":/opt/project/resources/jars/aimedic-grouper-assembly.jar "$(pwd)"/resources/jars/

echo "--- done ---"
