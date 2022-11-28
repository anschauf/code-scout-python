FROM eclipse-temurin:18.0.2.1_1-jre AS AWS-CLI

RUN apt-get update -y
RUN apt-get install unzip -y

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-aarch64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install -i ~/aws-cli -b ~/aws-cli/bin

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_REGION
ARG AIMEDIC_GROUPER_VERSION

ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ENV AWS_REGION=$AWS_REGION
ENV AIMEDIC_GROUPER_VERSION=$AIMEDIC_GROUPER_VERSION

RUN ~/aws-cli/bin/aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
RUN ~/aws-cli/bin/aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
RUN ~/aws-cli/bin/aws configure set default.region $AWS_REGION

RUN mkdir -p /tmp/jars
RUN ~/aws-cli/bin/aws codeartifact get-package-version-asset --domain aimedic --domain-owner 264427866130 --repository aimedic --format maven --namespace ch.aimedic --package aimedic-grouper_2.12 --package-version ${AIMEDIC_GROUPER_VERSION} --asset aimedic-grouper-assembly-${AIMEDIC_GROUPER_VERSION}.jar /tmp/jars/aimedic-grouper-assembly.jar


FROM arm64v8/python:3.10

RUN mkdir -p ./resources/jars
COPY --from=AWS-CLI /tmp/jars/aimedic-grouper-assembly.jar ./resources/jars

RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
COPY constraints.txt .
RUN pip3 install -r requirements.txt -c constraints.txt
