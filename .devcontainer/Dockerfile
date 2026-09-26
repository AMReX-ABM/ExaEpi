FROM nvidia/cuda:12.8.0-devel-ubuntu24.04

# Install build deps and Python packages
RUN apt-get update && apt-get upgrade -y && apt-get install -y --no-install-recommends \
    cmake git git-lfs g++ make ninja-build \
    libopenmpi-dev openmpi-bin libhdf5-openmpi-dev \
    ccache python3 python3-pip python3-venv clang-format sudo \
    curl ca-certificates poppler-utils time \
    && pip3 install --no-cache-dir --break-system-packages \
       numpy==2.5.3 matplotlib==3.11.2 pandas==3.0.5 pandas-stubs==3.0.5.260914 \
       h5py==3.16.0 scipy==1.18.1 geopandas==1.1.4 yt==4.4.2 \
       polars==1.44.2 pyarrow==25.0.1 psutil==7.2.2 colorama==0.4.6 \
       censusgeocode==0.5.3 dill==0.4.1 black==26.5.1 \
    && curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Versions are pinned to what the container was actually running when this was last updated, so
# a rebuild reproduces a known-working environment rather than whatever is newest that day. Bump
# them deliberately.
#
# black    -- Python formatter; was installed by hand with `pip install --user`, which a rebuild
#             would silently lose.
# pyarrow  -- polars needs it to convert to/from pandas, which several utilities/ scripts do.
# dill     -- UrbanPop ships data/UrbanPop/base/*/moe_fit_rates_*.pkl written with dill; reading
#             them needs it, or a shim that stands in for it.
# time     -- GNU /usr/bin/time; `time -v` reports peak RSS, which the UrbanPop experiments use to
#             measure solve memory. The bash builtin `time` does not.
# python3-venv -- provides ensurepip, without which `python3 -m venv` fails outright. Needed for
#             the isolated livelike/pymedm environment created below.

# Rename the existing ubuntu user/group (UID/GID 1000) to match USERNAME
ARG USERNAME=user
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupmod --new-name ${USERNAME} ubuntu \
    && usermod --login ${USERNAME} --move-home --home /home/${USERNAME} ubuntu \
    # Allow passwordless sudo
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} \
    && chmod 0440 /etc/sudoers.d/${USERNAME}

# Ensure ccache and Claude Code config directories are owned by the user
# (these get overlaid by named volumes in devcontainer.json, which seed their
# initial ownership from whatever exists here at first mount)
RUN mkdir -p /home/${USERNAME}/.ccache /home/${USERNAME}/.claude /home/${USERNAME}/.config/exaepi \
    && chown -R ${USER_UID}:${USER_GID} /home/${USERNAME}/.ccache /home/${USERNAME}/.claude \
                                        /home/${USERNAME}/.config/exaepi \
    && chmod 700 /home/${USERNAME}/.config/exaepi

# Isolated environment for UrbanPop population synthesis (livelike + pymedm, which pull in jax
# and jaxopt). Kept out of the system environment because that dependency tree is large and
# moves fast, and an upgrade there should not be able to disturb the numpy/scipy/polars versions
# the utilities/ analysis scripts run on.
#
# It lives in /opt, NOT in the workspace: /workspaces is a bind mount of the host checkout, so
# anything written there is not part of the image and would not survive a rebuild -- which is
# the whole point of putting it here.
#
# The solver stack is pinned exactly, because its behaviour is load-bearing and has been measured:
# jaxopt's L-BFGS default of maxiter=500 is what pymedm solves stop at, and pymedm's unregularised
# Hessian inversion is what build_precompute.py's draw_replicates works around. A silent upgrade
# of either would change bundles without anything failing.
#
# jax[cuda12] adds the GPU plugin (with its own CUDA libraries as pip wheels, several GB) so the
# dense-GEMM solver can be tested on the GPU. JAX would then default to the GPU, which is wrong for
# pymedm itself -- batch_solve threads across PUMAs on CPU but runs them sequentially on a GPU, and
# the problem is solved in float64, where consumer GPUs are slow -- so JAX_PLATFORMS=cpu keeps CPU
# the default. Set JAX_PLATFORMS=cuda explicitly to use the GPU.
#
# polars, psutil and colorama are imported by upop_to_exaepi.py, which build_precompute.py and
# generate_population.py load for their shared tables; pypdf reads the papers in related/.
ENV LIVELIKE_VENV=/opt/livelike-venv
ENV JAX_PLATFORMS=cpu
RUN python3 -m venv ${LIVELIKE_VENV} \
    && ${LIVELIKE_VENV}/bin/pip install --no-cache-dir --upgrade pip \
    && ${LIVELIKE_VENV}/bin/pip install --no-cache-dir \
       livelike==1.5.3 pymedm==2.2.8 \
       "jax[cuda12]==0.11.2" jaxlib==0.11.2 jaxopt==0.8.5 \
       polars==1.44.2 pyarrow==25.0.1 psutil==7.2.2 colorama==0.4.6 dill==0.4.1 \
       pypdf==6.19.0 \
    && chown -R ${USER_UID}:${USER_GID} ${LIVELIKE_VENV}

USER ${USERNAME}
