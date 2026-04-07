#!/usr/bin/env bash
#
# build-cirun-gpu-image.sh
#
# Builds a GCP custom image with NVIDIA GPU drivers + CUDA pre-installed,
# ready to use with Cirun for GPU CI.
#
# Approach: start from GCP Deep Learning VM (drivers already installed),
# create a disk image in YOUR project so Cirun can access it.
#
# Prerequisites:
#   - gcloud CLI installed and authenticated
#   - Sufficient GPU quota in target zone (NVIDIA_T4_GPUS >= 1)
#   - Compute Engine API enabled
#
# Usage:
#   GCP_PROJECT=my-project ./build-cirun-gpu-image.sh
#

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
GCP_PROJECT="${GCP_PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
GCP_ZONE="${GCP_ZONE:-asia-east1-a}"
IMAGE_NAME="${IMAGE_NAME:-cirun-nvidia-gpu}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-50}"

# Deep Learning VM image family (drivers + CUDA pre-installed)
# Project: deeplearning-platform-release
# To list available families:
#   gcloud compute images list --project=deeplearning-platform-release \
#     --no-standard-images --format="value(family)" | sort -u | grep common
DL_IMAGE_FAMILY="${DL_IMAGE_FAMILY:-common-cu128-ubuntu-2204-nvidia-570}"
DL_IMAGE_PROJECT="${DL_IMAGE_PROJECT:-deeplearning-platform-release}"

# Temporary VM name
TEMP_VM="cirun-image-builder-$(date +%s)"

# ─── Validation ───────────────────────────────────────────────────────────────
if [[ -z "$GCP_PROJECT" ]]; then
    echo "ERROR: GCP_PROJECT is not set and no default project configured."
    echo "  Run:  gcloud config set project YOUR_PROJECT_ID"
    echo "  Or:   GCP_PROJECT=your-project ./build-cirun-gpu-image.sh"
    exit 1
fi

echo "============================================="
echo " Cirun GPU Image Builder"
echo "============================================="
echo " Project:      $GCP_PROJECT"
echo " Zone:         $GCP_ZONE"
echo " Image name:   $IMAGE_NAME"
echo " DL VM family: $DL_IMAGE_FAMILY"
echo " Machine type: $MACHINE_TYPE"
echo " Disk size:    ${BOOT_DISK_SIZE}GB"
echo " Temp VM:      $TEMP_VM"
echo "============================================="
echo ""

# ─── Preflight: verify image family exists ───────────────────────────────────
echo ">>> Checking image family '${DL_IMAGE_FAMILY}' in project '${DL_IMAGE_PROJECT}' ..."
if ! gcloud compute images describe-from-family "$DL_IMAGE_FAMILY" \
    --project="$DL_IMAGE_PROJECT" --format="value(name)" &>/dev/null; then
    echo ""
    echo "ERROR: Image family '$DL_IMAGE_FAMILY' not found in project '$DL_IMAGE_PROJECT'."
    echo ""
    echo "Available common GPU families:"
    echo "  Try one of these:"
    echo "    common-cu128-ubuntu-2204-nvidia-570  (latest, CUDA 12.8)"
    echo "    pytorch-2-7-cu128-ubuntu-2204-nvidia-570  (with PyTorch)"
    echo ""
    echo "  Or discover families with:"
    echo "    gcloud compute images describe-from-family FAMILY_NAME \\"
    echo "      --project=deeplearning-platform-release --format='value(name)'"
    echo ""
    echo "Re-run with:  DL_IMAGE_FAMILY=<family-from-above> $0"
    exit 1
fi
RESOLVED_IMAGE=$(gcloud compute images describe-from-family "$DL_IMAGE_FAMILY" \
    --project="$DL_IMAGE_PROJECT" --format="value(name)")
echo "    Resolved to: $RESOLVED_IMAGE"
echo ""

# ─── Check if image already exists ───────────────────────────────────────────
if gcloud compute images describe "$IMAGE_NAME" --project="$GCP_PROJECT" &>/dev/null; then
    echo "WARNING: Image '$IMAGE_NAME' already exists in project '$GCP_PROJECT'."
    read -rp "Delete and rebuild? [y/N] " confirm
    if [[ "$confirm" =~ ^[yY]$ ]]; then
        echo ">>> Deleting existing image ..."
        gcloud compute images delete "$IMAGE_NAME" --project="$GCP_PROJECT" --quiet
    else
        echo "Aborted."
        exit 0
    fi
fi

# ─── Cleanup trap ─────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo ">>> Cleaning up temporary VM ..."
    gcloud compute instances delete "$TEMP_VM" \
        --project="$GCP_PROJECT" \
        --zone="$GCP_ZONE" \
        --quiet 2>/dev/null || true
}
trap cleanup EXIT

# ─── Step 1: Create temporary VM from Deep Learning VM image ─────────────────
echo ">>> Step 1/4: Creating temporary VM from Deep Learning VM image ..."
echo "    (image family: $DL_IMAGE_FAMILY from project: $DL_IMAGE_PROJECT)"
echo ""

gcloud compute instances create "$TEMP_VM" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --machine-type="$MACHINE_TYPE" \
    --accelerator="type=nvidia-tesla-t4,count=1" \
    --maintenance-policy=TERMINATE \
    --image-family="$DL_IMAGE_FAMILY" \
    --image-project="$DL_IMAGE_PROJECT" \
    --boot-disk-size="${BOOT_DISK_SIZE}GB" \
    --boot-disk-type=pd-balanced \
    --metadata=install-nvidia-driver=True \
    --no-restart-on-failure

echo ""
echo ">>> Step 2/4: Waiting for VM to be ready and drivers to initialize ..."
echo "    (this may take 2-5 minutes)"

# Wait for SSH to become available
for i in $(seq 1 30); do
    if gcloud compute ssh "$TEMP_VM" \
        --project="$GCP_PROJECT" \
        --zone="$GCP_ZONE" \
        --command="echo ready" \
        --ssh-flag="-o ConnectTimeout=10" \
        --ssh-flag="-o StrictHostKeyChecking=no" \
        2>/dev/null; then
        break
    fi
    echo "    Waiting for SSH... (attempt $i/30)"
    sleep 10
done

# ─── Step 3: Verify GPU drivers work ─────────────────────────────────────────
echo ""
echo ">>> Step 3/4: Verifying GPU drivers on temporary VM ..."

gcloud compute ssh "$TEMP_VM" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --ssh-flag="-o StrictHostKeyChecking=no" \
    --command="nvidia-smi"

echo ""
echo "    GPU verified successfully."

# ─── Step 4: Stop VM and create image ────────────────────────────────────────
echo ""
echo ">>> Step 4/4: Stopping VM and creating image ..."

gcloud compute instances stop "$TEMP_VM" \
    --project="$GCP_PROJECT" \
    --zone="$GCP_ZONE" \
    --quiet

echo "    VM stopped. Creating image '${IMAGE_NAME}' ..."

gcloud compute images create "$IMAGE_NAME" \
    --project="$GCP_PROJECT" \
    --source-disk="$TEMP_VM" \
    --source-disk-zone="$GCP_ZONE" \
    --family="cirun-gpu" \
    --description="Cirun GPU runner image with NVIDIA drivers + CUDA (built $(date -u +%Y-%m-%d))"

echo ""
echo "============================================="
echo " Image built successfully!"
echo "============================================="
echo ""
echo " Image:   ${IMAGE_NAME}"
echo " Family:  cirun-gpu"
echo " Project: ${GCP_PROJECT}"
echo ""
echo " .cirun.yml config:"
echo ""
echo "   runners:"
echo "     - name: gpu-runner"
echo "       cloud: gcp"
echo "       gpu: nvidia-tesla-t4"
echo "       instance_type: n1-standard-4"
echo "       machine_image: \"${GCP_PROJECT}:${IMAGE_NAME}\""
echo "       region: \"${GCP_ZONE}\""
echo "       preemptible: false"
echo "       labels:"
echo "         - cirun-gpu-runner"
echo ""
