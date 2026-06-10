# Patched amica17 Fortran reference, portable build.
#
# Build:
#   docker build -t amica17 .
#
# Run:
#   docker run --rm -v $PWD:/work -w /work amica17 amica17 myparam.param
#
# The image contains:
#   - /usr/local/bin/amica17  (the binary)
#   - /usr/local/share/amica/  (the source for re-compilation if needed)

FROM ubuntu:22.04 AS build

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    gfortran \
    libopenmpi-dev \
    openmpi-bin \
    libopenblas-dev \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY amica17_patched.f90 amica17_header.f90 funmod2.f90 mkl_stubs.f90 mkl_vml.f90 build.sh ./
RUN chmod +x build.sh && BLAS=openblas ./build.sh

FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgfortran5 \
    libgomp1 \
    libopenmpi3 \
    libopenblas0 \
    openmpi-bin \
 && rm -rf /var/lib/apt/lists/*

COPY --from=build /src/amica17 /usr/local/bin/amica17
COPY --from=build /src/amica17_patched.f90 /usr/local/share/amica/
COPY --from=build /src/amica17_header.f90 /usr/local/share/amica/
COPY --from=build /src/funmod2.f90 /usr/local/share/amica/
COPY --from=build /src/mkl_stubs.f90 /usr/local/share/amica/
COPY --from=build /src/build.sh /usr/local/share/amica/

CMD ["amica17"]
