# set base image (host OS)
FROM python:3.8

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN apt-get update && apt-get -y install --no-install-recommends \
	gcc \
	g++ \
	gfortran \
	libopenblas-dev \
	libblas-dev \
	liblapack-dev \
	libatlas-base-dev \
	libhdf5-dev \
	libhdf5-103 \
	pkg-config \
	python3-dev \
	python3-pip \
	python3-setuptools \
	pybind11-dev \
	wget
RUN pip install -r requirements.txt

ENV OPENCV_VERSION=4.5.0

# building open-cv
WORKDIR /root
RUN wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip
RUN wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip
RUN unzip -o $OPENCV_VERSION.zip && unzip opencv_contrib.zip && mv opencv_contrib-$OPENCV_VERSION opencv_contrib
RUN apt-get -y install cmake
WORKDIR /root/opencv-$OPENCV_VERSION/build

RUN cmake -D ENABLE_NEON=ON  -D ENABLE_VFPV3=ON \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
      -D WITH_FFMPEG=ON -D WITH_TBB=ON -D WITH_GTK=ON -D WITH_V4L=ON -D WITH_OPENGL=ON -D WITH_CUBLAS=ON -DWITH_QT=OFF \
      -DCUDA_NVCC_FLAGS="-D_FORCE_INLINES"  --prefix=/usr --extra-version='1~16.04.york0' --toolchain=hardened --libdir=/usr/lib/x86_64-linux-gnu \
      --incdir=/usr/include/x86_64-linux-gnu --arch=amd64 --enable-gpl --disable-stripping --enable-avresample --disable-filter=resample \
      --enable-avisynth --enable-gnutls --enable-ladspa --enable-libaom --enable-libass --enable-libbluray --enable-libbs2b --enable-libcaca \
      --enable-libcdio --enable-libcodec2 --enable-libflite --enable-libfontconfig --enable-libfreetype --enable-libfribidi --enable-libgme \
      --enable-libgsm --enable-libjack --enable-libmp3lame --enable-libmysofa --enable-libopenjpeg --enable-libopenmpt --enable-libopus \
      --enable-libpulse --enable-librsvg --enable-librubberband --enable-libshine --enable-libsnappy --enable-libsoxr --enable-libspeex \
      --enable-libssh --enable-libtheora --enable-libtwolame --enable-libvidstab --enable-libvorbis --enable-libvpx --enable-libwavpack \
      --enable-libwebp --enable-libx265 --enable-libxml2 --enable-libxvid --enable-libzmq --enable-libzvbi --enable-lv2 --enable-omx \
      --enable-openal --enable-opengl --enable-sdl2 --enable-libdc1394 --enable-libdrm --enable-libiec61883 --enable-chromaprint --enable-frei0r \
      --enable-libopencv --enable-libx264 --enable-shared ..

# set the working directory in the container
WORKDIR /root/parking_detect


# command to run on container start
CMD [ "python", "./training/eval_img_parking_lite.py" ]