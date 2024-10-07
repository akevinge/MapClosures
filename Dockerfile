FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y && \
    apt install --no-install-recommends -y \
    python3 \
    python3-pip \
    python3.10-dev \
    # Build dependencies found here: https://github.com/PRBonn/MapClosures/blob/main/README.md#Install.
    build-essential \
    cmake \
    pybind11-dev \
    libeigen3-dev \
    libopencv-dev \
    libtbb-dev

# Runtime dependencies found here: https://github.com/PRBonn/MapClosures/blob/main/README.md#Install.
RUN pip install \
    kiss-icp==1.0.0 \
    rosbags==0.10.4 \
    matplotlib==3.9.2

COPY docker/entrypoint.sh /opt/entrypoint.sh

RUN mkdir -p /map_closures

WORKDIR /map_closures

ENTRYPOINT ["/opt/entrypoint.sh"]