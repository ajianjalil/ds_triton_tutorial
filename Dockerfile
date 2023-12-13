FROM nvcr.io/nvidia/deepstream:6.3-triton-multiarch
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-get update
RUN apt-get install -y python3-pip \
    cmake 
RUN python3 -m pip install scikit-build
RUN python3 -m pip install numpy
RUN python3 -m pip install opencv-python
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
    python3-numpy python3-opencv
WORKDIR /opt/nvidia/deepstream/deepstream-6.3/
