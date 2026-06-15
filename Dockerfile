# Dockerfile
FROM ubuntu:22.04

# --- base system deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion ca-certificates curl git wget vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*


# --- Miniconda ---
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p ${CONDA_DIR} \
    && rm -f /tmp/miniconda.sh
ENV PATH=${CONDA_DIR}/bin:$PATH
RUN ln -s ${CONDA_DIR}/etc/profile.d/conda.sh /etc/profile.d/conda.sh
SHELL ["/bin/bash", "-lc"]

# --- Conda env (Python 3.7) ---
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
RUN conda update -y -n base -c defaults conda \
    && conda create -y -n hpo python=3.7 \
    && echo 'source /etc/profile.d/conda.sh' >> ~/.bashrc \
    && echo 'conda activate hpo' >> ~/.bashrc

# --- HPOBench preparation ---
COPY HPOBench /opt/HPOBench

RUN source /etc/profile.d/conda.sh && conda activate hpo \
    && pip install -U pip setuptools wheel ipython
RUN source /etc/profile.d/conda.sh && conda activate hpo \
    && pip install -e /opt/HPOBench

WORKDIR /workspace

EXPOSE 8000

CMD ["/bin/bash"]
