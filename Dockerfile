FROM ubuntu:24.04 AS builder

# Get dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    g++ \
    make \
    wget \
    unzip \
    libgtk2.0-dev \
    pkg-config


# build and install opencv. Might take 10 mins or more
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.11.0.zip
RUN unzip opencv.zip
RUN mv opencv-4.11.0 opencv
WORKDIR  /opencv
RUN mkdir build
WORKDIR  /opencv/build
RUN cmake ../
RUN make -j4
RUN make install

# build the thing
COPY ./MatchTool /MatchTool
WORKDIR /MatchTool
RUN ./build.sh

# Do whatever you want now
ENV LD_LIBRARY_PATH=/usr/local/lib
CMD ["./out/main"]