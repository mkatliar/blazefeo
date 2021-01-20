FROM ubuntu:groovy
WORKDIR /root
RUN apt-get update
RUN apt-get upgrade -y
RUN DEBIAN_FRONTEND="noninteractive" apt-get install -y \
    build-essential clang-11 cmake git libopenblas-dev libboost-exception-dev pkg-config python3-matplotlib

# Install GTest
RUN git clone https://github.com/google/googletest.git
RUN cd googletest && cmake -DCMAKE_BUILD_TYPE=Release . && make -j `nproc` install

# Install Google benchmark
RUN git clone https://github.com/google/benchmark.git
RUN cd benchmark && cmake -DCMAKE_BUILD_TYPE=Release -DBENCHMARK_ENABLE_GTEST_TESTS=False . && make -j `nproc` install

# Install Blaze
RUN git clone https://bitbucket.org/blaze-lib/blaze.git
RUN cd blaze && cmake -DBLAZE_BLAS_MODE=True -DBLAZE_BLAS_USE_MATRIX_MATRIX_MULTIPLICATION=False \
    -DBLAZE_BLAS_USE_MATRIX_VECTOR_MULTIPLICATION=False -DBLAZE_VECTORIZATION=True -DCMAKE_INSTALL_PREFIX=/usr/local/ . && make install

# Install Eigen3
RUN git clone https://gitlab.com/libeigen/eigen.git
RUN mkdir -p eigen/build && cd eigen/build && cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ .. && make install

# Install blasfeo
RUN apt-get install -y bc
RUN git clone https://github.com/giaf/blasfeo.git
RUN cd blasfeo && git checkout cc90e146ee9089de518f57dbb736e064bd82394e
COPY docker/blasfeo/Makefile.rule blasfeo
RUN cd blasfeo && make -j `nproc` static_library && make install_static

# Install libxsmm
RUN git clone https://github.com/hfp/libxsmm.git
RUN cd libxsmm && make -j `nproc` PREFIX=/usr/local install

# Install MKL
RUN apt-get install -y wget
RUN wget https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN apt-key add GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN rm GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN wget https://apt.repos.intel.com/setup/intelproducts.list -O /etc/apt/sources.list.d/intelproducts.list
RUN apt-get update
RUN apt-get install -y intel-mkl-64bit-2020.4-912

# Build blazefeo
COPY bench blazefeo/bench
COPY cmake blazefeo/cmake
COPY include blazefeo/include
COPY test blazefeo/test
COPY CMakeLists.txt blazefeo
COPY Makefile blazefeo/Makefile
ENV PKG_CONFIG_PATH=/usr/local/lib
RUN mkdir -p blazefeo/build && cd blazefeo/build \
    && cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo \
        -DCMAKE_CXX_COMPILER="clang++-11" \
        -DCMAKE_CXX_FLAGS="-march=native -mfma -mavx -mavx2 -msse4 -fno-math-errno" .. \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -g -DNDEBUG -ffast-math" \
    && make -j `nproc` VERBOSE=1

# Run tests
RUN cd blazefeo/build && ctest

# Run benchmarks
ENV MKL_THREADING_LAYER=SEQUENTIAL
ENV OPENBLAS_NUM_THREADS=1
CMD mkdir -p blazefeo/bench_result/data \
    && mkdir -p blazefeo/bench_result/image \
    && cd blazefeo \
    && make -j 1 bench_result/image/dgemm_performance.png bench_result/image/dgemm_performance_ratio.png

