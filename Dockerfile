FROM amazon/aws-cli AS AWS-CLI

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_REGION
ARG AIMEDIC_GROUPER_VERSION

USER root
RUN yum update
RUN yum install -y java

RUN aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
RUN aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
RUN aws configure set default.region $AWS_REGION


RUN mkdir "/tmp/jars"
RUN aws codeartifact get-package-version-asset --domain aimedic --domain-owner 264427866130 --repository aimedic --format maven --namespace ch.aimedic --package aimedic-grouper_2.12 --package-version ${AIMEDIC_GROUPER_VERSION} --asset aimedic-grouper-assembly-${AIMEDIC_GROUPER_VERSION}.jar /tmp/jars/aimedic-grouper-assembly.jar

FROM jupyter/datascience-notebook:aarch64-lab-3.4.7

RUN mkdir -p /tmp/jars
COPY --from=AWS-CLI /tmp/jars/aimedic-grouper-assembly.jar /tmp/jars

WORKDIR "/home/jovyan/work"
USER root
RUN mkdir -p ./resources/jars

RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
COPY constraints.txt .
RUN pip3 install -r requirements.txt -c constraints.txt
