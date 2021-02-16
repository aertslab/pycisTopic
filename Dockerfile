FROM python:3.8-slim AS compile-image

ENV DEBIAN_FRONTEND=noninteractive
RUN BUILDPKGS="build-essential \
        libcurl4-openssl-dev \
        zlib1g-dev \
        libfftw3-dev \
        libc++-dev \
        git \
        wget \
        " && \
    apt-get update && \
    apt-get install -y --no-install-recommends apt-utils debconf locales && dpkg-reconfigure locales && \
    apt-get install -y --no-install-recommends $BUILDPKGS

RUN python -m venv /opt/venv
# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

# install dependencies:
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir --upgrade pip wheel && \
    pip install --no-cache-dir Cython numpy && \
    pip install --no-cache-dir fitsne && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# install pycisTopic from local copy for now:
COPY . /tmp/pycisTopic
RUN  cd /tmp/pycisTopic && \
     pip install . && \
     cd .. && rm -rf pycisTopic


FROM python:3.8-slim AS build-image

RUN apt-get -y update && \
    apt-get -y --no-install-recommends install \
        # Need to run ps
        procps \
        bash-completion \
        less && \
    rm -rf /var/cache/apt/* && \
    rm -rf /var/lib/apt/lists/*

COPY --from=compile-image /opt/venv /opt/venv

# Make sure we use the virtualenv:
ENV PATH="/opt/venv/bin:$PATH"

