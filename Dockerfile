FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update
RUN apt-get install -y python3-pip \
    cmake 
RUN python3 -m pip install scikit-build
RUN python3 -m pip install numpy
RUN apt-get install -y gstreamer-1.0 \
     gir1.2-gst-rtsp-server-1.0  \
     python3-gi \
     iputils-ping \
     python3-gst-1.0 \
     libgstreamer1.0-dev \
     libgstreamer-plugins-base1.0-dev \
     cmake \
     pkg-config
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gir1.2-gst-rtsp-server-1.0

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gstreamer1.0-rtsp
RUN apt install libgl1-mesa-glx -y
RUN apt-get install 'ffmpeg'\
    'libsm6'\
    'libxext6'  -y
RUN apt-get install -y libgirepository1.0-dev \
    gobject-introspection gir1.2-gst-rtsp-server-1.0 \
    python3-numpy

RUN python3 -m pip install pyds_ext
RUN python3 -m pip install cupy==12.3.0



ARG DEBIAN_FRONTEND=noninteractive
ARG OPENCV_VERSION=4.9.0

RUN apt-get update && \
    # Install build tools, build dependencies and python
    apt-get install -y \
	python3-pip \
        build-essential \
        cmake \
        git \
        wget \
        unzip \
        yasm \
        pkg-config \
        libswscale-dev \
        libtbb2 \
        libtbb-dev \
        libjpeg-dev \
        libpng-dev \
        libtiff-dev \
        libavformat-dev \
        libpq-dev \
        libxine2-dev \
        libglew-dev \
        libtiff5-dev \
        zlib1g-dev \
        libjpeg-dev \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libpostproc-dev \
        libswscale-dev \
        libeigen3-dev \
        libtbb-dev \
        libgtk2.0-dev \
        pkg-config \
        ## Python
        python3-dev \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

RUN cd /opt/ &&\
    # Download and unzip OpenCV and opencv_contrib and delte zip files
    wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip &&\
    unzip $OPENCV_VERSION.zip &&\
    rm $OPENCV_VERSION.zip &&\
    wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip &&\
    unzip ${OPENCV_VERSION}.zip &&\
    rm ${OPENCV_VERSION}.zip &&\
    # Create build folder and switch to it
    mkdir /opt/opencv-${OPENCV_VERSION}/build && cd /opt/opencv-${OPENCV_VERSION}/build &&\
    # Cmake configure
    cmake \
        -DOPENCV_EXTRA_MODULES_PATH=/opt/opencv_contrib-${OPENCV_VERSION}/modules \
        -DWITH_CUDA=ON \
        -DCUDA_ARCH_BIN=7.5,8.0,8.6 \
        -DCMAKE_BUILD_TYPE=RELEASE \
        # Install path will be /usr/local/lib (lib is implicit)
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        .. &&\
    # Make
    make -j"$(nproc)" && \
    # Install to /usr/local/lib
    make install && \
    ldconfig &&\
    # Remove OpenCV sources and build folder
    rm -rf /opt/opencv-${OPENCV_VERSION} && rm -rf /opt/opencv_contrib-${OPENCV_VERSION}

ENV PYTHONPATH=/usr/local/lib/python3.8/site-packages/:$PYTHONPATH
WORKDIR /opt/nvidia/deepstream/deepstream-6.3/sources/deepstream_python_apps/apps/deepstream-rtsp-in-rtsp-out/
