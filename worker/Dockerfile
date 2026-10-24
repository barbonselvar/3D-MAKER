# 3D-MAKER GPU worker image: TRELLIS.2-4B + validation tools (blueprint P1-02).
# STATUS: NOT VERIFIED - written without access to a GPU or a Docker daemon.
#
# Build (from the repository root, on a Linux host with the NVIDIA container toolkit):
#   docker build -f worker/Dockerfile -t 3d-maker-worker:dev .
# Smoke test (>= 24 GB NVIDIA GPU; downloads ~16 GB of weights into the cache volume):
#   docker run --rm --gpus all -e HF_TOKEN -e TRELLIS2_WEIGHTS_REVISION=<hf commit> \
#     -v maker-models:/models -v "$PWD/out:/out" 3d-maker-worker:dev --smoke
# Then pin the result: record `docker inspect --format '{{index .RepoDigests 0}}'` in worker/README.md.

# nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04, pinned by manifest-list digest (Docker Hub, 2024-04-24).
# CUDA 12.4 + PyTorch 2.6 is the pair TRELLIS.2's setup.sh installs; Ubuntu 22.04 ships Python 3.10.
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:622e78a1d02c0f90ed900e3985d6c975d8e2dc9ee5e61643aed587dcf9129f42

# A100/A10/L40S/H100. Add 8.6 for RTX 30xx / A6000 if needed (longer build).
ARG TORCH_CUDA_ARCH_LIST="8.0;8.9;9.0"
ARG TRELLIS2_COMMIT=75fbf0183001ed9876c8dbb35de6b68552ee08bd
ARG NVDIFFRAST_COMMIT=253ac4fcea7de5f396371124af597e6cc957bfae
ARG NVDIFFREC_COMMIT=b296927cc7fd01c2ac1087c8065c4d7248f72da4
ARG CUMESH_COMMIT=12289e1062f0603f2f0d0771b02e1395d247f26f
ARG FLEXGEMM_COMMIT=6dd94a859c26ee8246888502eada3dd8ad85532e
ARG UTILS3D_COMMIT=9a4eb15e4021b67b12c460c7057d642626897ec8
ARG BLENDER_SERIES=4.2
ARG BLENDER_VERSION=4.2.3
ARG NODE_VERSION=22.11.0
ARG NODE_SHA256=83bf07dd343002a26211cf1fcd46a9d9534219aad42ee02847816940bf610a72

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} MAX_JOBS=8 PIP_CONSTRAINT=/opt/maker/worker/constraints-gpu.txt

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-dev python3-pip python-is-python3 git ca-certificates curl xz-utils \
        build-essential ninja-build libjpeg-dev libgl1 libglib2.0-0 libegl1 libxi6 libxrender1 libxkbcommon0 libsm6 \
    && rm -rf /var/lib/apt/lists/*

COPY worker/constraints-gpu.txt /opt/maker/worker/constraints-gpu.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
    && pip install imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja trimesh transformers \
        huggingface_hub safetensors pandas lpips zstandard plyfile kornia timm \
    && pip install "git+https://github.com/EasternJournalist/utils3d.git@${UTILS3D_COMMIT}" \
    && pip install flash-attn --no-build-isolation

# CUDA extensions, each at an exact commit (upstream setup.sh clones branch heads).
RUN set -eux; mkdir -p /ext; cd /ext; \
    fetch() { git clone --filter=blob:none "$1" "$2" && git -C "$2" checkout "$3" && git -C "$2" submodule update --init --recursive; }; \
    fetch https://github.com/NVlabs/nvdiffrast.git nvdiffrast "${NVDIFFRAST_COMMIT}"; \
    fetch https://github.com/JeffreyXiang/nvdiffrec.git nvdiffrec "${NVDIFFREC_COMMIT}"; \
    fetch https://github.com/JeffreyXiang/CuMesh.git CuMesh "${CUMESH_COMMIT}"; \
    fetch https://github.com/JeffreyXiang/FlexGEMM.git FlexGEMM "${FLEXGEMM_COMMIT}"; \
    fetch https://github.com/microsoft/TRELLIS.2.git /opt/trellis2 "${TRELLIS2_COMMIT}"; \
    for pkg in nvdiffrast nvdiffrec CuMesh FlexGEMM /opt/trellis2/o-voxel; do pip install "$pkg" --no-build-isolation; done; \
    rm -rf /ext

# Blender LTS for validation gate V2, verified against blender.org's published SHA-256 list.
RUN set -eux; cd /tmp; \
    base="https://download.blender.org/release/Blender${BLENDER_SERIES}"; \
    curl -fsSLO "${base}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"; \
    curl -fsSL "${base}/blender-${BLENDER_VERSION}.sha256" | grep "linux-x64.tar.xz" | sha256sum -c -; \
    mkdir -p /opt/blender && tar -xJf "blender-${BLENDER_VERSION}-linux-x64.tar.xz" -C /opt/blender --strip-components=1; \
    rm "blender-${BLENDER_VERSION}-linux-x64.tar.xz"

# Node.js for the Khronos glTF-Validator (gate V1).
RUN set -eux; cd /tmp; \
    curl -fsSLO "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz"; \
    echo "${NODE_SHA256}  node-v${NODE_VERSION}-linux-x64.tar.xz" | sha256sum -c -; \
    tar -xJf "node-v${NODE_VERSION}-linux-x64.tar.xz" -C /usr/local --strip-components=1; rm node-*.tar.xz

WORKDIR /opt/maker
COPY worker/requirements.txt worker/package.json worker/package-lock.json worker/
RUN pip install -r worker/requirements.txt && npm ci --prefix worker --omit=dev --no-audit --no-fund
COPY app/__init__.py app/
COPY app/core/__init__.py app/core/engines.py app/core/
COPY worker/ worker/

# Non-root, no inbound ports; weights cache on a volume.
RUN useradd --uid 10001 --create-home maker && mkdir -p /models /out && chown maker /models /out
USER maker
ENV PYTHONPATH=/opt/maker:/opt/trellis2 HF_HOME=/models BLENDER_BIN=/opt/blender/blender \
    VALIDATION_REQUIRE_BLENDER=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    OPENCV_IO_ENABLE_OPENEXR=1 WORKER_ENGINES=trellis2-4b SEGMENTER=birefnet
ENTRYPOINT ["python", "-m", "worker.main"]
