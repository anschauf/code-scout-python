FROM arm64v8/python:3.10.7

ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ARG AWS_REGION
ARG AIMEDIC_GROUPER_VERSION

ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV AWS_REGION=${AWS_REGION}
ENV AIMEDIC_GROUPER_VERSION=${AIMEDIC_GROUPER_VERSION}

RUN apt-get update
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install awscli --force-reinstall --upgrade

RUN aws configure set aws_access_key_id ${AWS_ACCESS_KEY_ID}
RUN aws configure set aws_secret_access_key ${AWS_SECRET_ACCESS_KEY}
RUN aws configure set default.region ${AWS_REGION}


RUN mkdir "/tmp/jars"
RUN aws codeartifact get-package-version-asset \
    --domain aimedic \
    --domain-owner 264427866130  \
    --repository aimedic  \
    --format maven  \
    --namespace ch.aimedic  \
    --package aimedic-grouper_2.12  \
    --package-version ${AIMEDIC_GROUPER_VERSION}  \
    --asset aimedic-grouper-assembly-${AIMEDIC_GROUPER_VERSION}.jar  \
    /tmp/jars/aimedic-grouper-assembly.jar


#FROM jupyter/datascience-notebook
#
#RUN echo $USER
#
#USER root
##RUN mkdir -p "/tmp/jars"
##RUN chown -R root:root /tmp/jars
#
#
#WORKDIR "/home/jovyan/work"
#RUN mkdir -p "./resources/jars"
#RUN chown -R root:root /home/jovyan/work
#
#COPY --from=AWS-CLI /tmp/jars/aimedic-grouper-assembly.jar /resources/jars
#
##RUN python3 -m pip install --upgrade pip
#COPY requirements.txt .
#COPY constraints.txt .
#
#RUN #find / -uid 100 -ls
#
#RUN pip3 install -r requirements.txt -c constraints.txt
#RUN #'/usr/local/bin/start.sh' jupyter lab --ServerApp.token=''

