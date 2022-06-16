FROM nvcr.io/nvidia/cuda:11.7.0-runtime-ubuntu20.04 as conda-base

RUN apt-get -qq update && apt-get -qq -y install curl bzip2 \
    && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -bfp /usr/local \
    && rm -rf /tmp/miniconda.sh \
    && conda install -y python=3 \
    && conda update conda \
    && apt-get -qq -y remove curl bzip2 \
    && apt-get -qq -y autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* /var/log/dpkg.log \
    && conda clean --all --yes

ENV PATH /opt/conda/bin:$PATH

FROM conda-base as torch-conda-env
ADD environment.yml /tmp/environment.yml
RUN conda env create torch-env -q -f /tmp/environment.yml
RUN conda init bash
RUN echo "conda activate torch-env" >> /root/.bashrc
ENV PATH /usr/local/envs/torch-env/bin:/usr/local/condabin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

FROM torch-conda-env as builder
COPY . /build_dir
WORKDIR /build_dir

RUN rm -rf dist
RUN python3 setup.py sdist bdist_wheel
RUN echo $(basename $(ls dist/vanilla_proteinmpnn*.whl)) > wheel_version
RUN cp dist/vanilla_proteinmpnn*.whl ./vanilla_proteinmpnn-latest.whl

FROM torch-conda-env as exec
COPY --from=builder /build_dir/vanilla_proteinmpnn-latest.whl /var/vanilla_proteinmpnn-latest.whl
COPY --from=builder /build_dir/wheel_version /var/wheel_version
RUN mv /var/vanilla_proteinmpnn-latest.whl /var/$(cat /var/wheel_version)
RUN pip install /var/$(cat /var/wheel_version)
ENV KMP_DUPLICATE_LIB_OK TRUE
ENV HDF5_USE_FILE_LOCKING FALSE
