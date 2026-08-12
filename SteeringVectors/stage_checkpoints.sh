#!/usr/bin/env bash
# Stage an OLMo-core checkpoint into teams/.../runs/ for GPU platform jobs.
#
# Platform batch roles cannot list or read edullm-checkpoints. Run this locally
# via sb_aws (or another role with read access to the source bucket) before
# submitting platform-run-vector-smoke.yaml.
#
# Example (370M smoke):
#   ./SteeringVectors/stage_checkpoints.sh \
#     s3://edullm-checkpoints/olmo-370m/edullm-370M-refhq-5p5b/checkpoints/step1315/ \
#     s3://sbsandbox-intern-edullm-outputs/teams/eval-inference/runs/steering-smoke/step1315/
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "Usage: $0 <source-s3-uri> <dest-s3-uri>" >&2
  exit 2
fi

source="${1%/}/"
dest="${2%/}/"

echo "Syncing ${source} -> ${dest}"
aws s3 sync "${source}" "${dest}" --only-show-errors
echo "Done. Verify:"
aws s3 ls "${dest}" --summarize
