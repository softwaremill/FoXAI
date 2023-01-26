FROM nvidia/cuda:11.6.2-devel-ubuntu20.04

RUN apt-get -y update; apt-get -y install curl
RUN apt-get -y install git
ADD nodejs.sh /
RUN chmod +x /nodejs.sh
RUN /nodejs.sh

#RUN mkdir -p /usr/bin
#RUN cp /usr/lib/wsl/lib/nvidia-smi /usr/bin/nvidia-smi 
#RUN chmod +x /usr/bin/nvidia-smi


RUN /bin/bash
