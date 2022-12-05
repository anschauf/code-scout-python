# The base image uses a Java Runtime based on Eclipse Temurin 18, which is the same we are currently using to build
# our Scala projects. Because we are downloading one of them here, we want to make sure to use pretty much the same
# environment, to avoid binary incompatibilities.


# -----------------------------------------------------------------------------
# Install the AWS CLI and download the grouper JAR
# Note: This is done in a separate container to avoid a leak of the environment variables and the AWS CLI itself into
# the runtime environment.
# -----------------------------------------------------------------------------
FROM eclipse-temurin:18.0.2.1_1-jre-jammy AS AWS-CLI

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_REGION

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_REGION=$AWS_REGION

# -----------------------------------------------------------------------------
# The grouper version is hard-coded here, so that it can be versioned with git
ENV AIMEDIC_GROUPER_VERSION=2.0.0_rc5
# -----------------------------------------------------------------------------

RUN apt-get update -y
RUN apt-get install unzip -y # needed to unzip the AWS CLI

# Install the AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install -i ~/aws-cli -b ~/aws-cli/bin

# Configure the AWS CLI to connect with privileges
RUN ~/aws-cli/bin/aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
RUN ~/aws-cli/bin/aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
RUN ~/aws-cli/bin/aws configure set default.region $AWS_REGION

# Download the aimedic-grouper JAR
RUN mkdir -p /tmp/jars
RUN ~/aws-cli/bin/aws codeartifact get-package-version-asset --domain aimedic --domain-owner 264427866130 --repository aimedic --format maven --namespace ch.aimedic --package aimedic-grouper_2.12 \
    --package-version ${AIMEDIC_GROUPER_VERSION} \
    --asset aimedic-grouper-assembly-${AIMEDIC_GROUPER_VERSION}.jar \
    /tmp/jars/aimedic-grouper-assembly.jar


# -----------------------------------------------------------------------------
# The runtime environment, containing python and our compiled Scala projects.
# Note: It seems easier to download python on a Docker image containing Java, than the other way round.
# -----------------------------------------------------------------------------
FROM eclipse-temurin:18.0.2.1_1-jre-jammy AS RUNTIME

# Install python
RUN apt-get update -y
RUN apt-get install -y --no-install-recommends ca-certificates curl python3.10 python3-pip python3-dev python3-setuptools python3-wheel

# Link python3 to python (https://askubuntu.com/q/320996)
RUN apt-get install python-is-python3 -y

# Copy the aimedic-grouper JAR from the other container
RUN mkdir -p /opt/project/resources/jars
WORKDIR /opt/project/resources/jars
COPY --from=AWS-CLI /tmp/jars/aimedic-grouper-assembly.jar .

# Install the python dependencies for code-scout-python
RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
COPY constraints.txt .
RUN pip3 install -r requirements.txt -c constraints.txt
