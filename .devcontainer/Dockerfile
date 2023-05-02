FROM continuumio/miniconda3:22.11.1

COPY spec-file.txt /tmp/conda-tmp/

RUN conda create \
    --name ript \
    --file /tmp/conda-tmp/spec-file.txt \
  && rm -rf /tmp/conda-tmp

ARG USERNAME=ript
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd \
    --gid $USER_GID \
    $USERNAME \
  && useradd \
    --uid $USER_UID \
    --gid $USER_GID \
    -m $USERNAME \
    #
    # [Optional] Add sudo support. Omit if you don't need to install software after connecting.
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
  && chown -R $USERNAME:$USERNAME /opt/conda/envs/ript


# # [Optional] Uncomment to install a different version of Python than the default
# # RUN conda install -y python=3.6 \
# #     && pip install --no-cache-dir pipx \
# #     && pipx reinstall-all

# # [Optional] Uncomment this section to install additional OS packages.
# # RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
# #     && apt-get -y install --no-install-recommends <your-package-list-here>

# # [Optional] Set the default user. Omit if you want to keep the default as root.
# USER $USERNAME
RUN apt-get update \
  && apt-get install -y\
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6
USER ript
