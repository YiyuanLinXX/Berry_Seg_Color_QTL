#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo_root/external-dependencies.lock"
mkdir -p "$repo_root/external"

checkout_dependency() {
  local name="$1"
  local url="$2"
  local commit="$3"
  local destination="$repo_root/external/$name"

  if [[ ! -d "$destination/.git" ]]; then
    git clone "$url" "$destination"
  fi
  git -C "$destination" fetch --tags origin
  git -C "$destination" checkout --detach "$commit"
}

checkout_dependency "SAM-CLIP" "$SAM_CLIP_URL" "$SAM_CLIP_COMMIT"
checkout_dependency "FoundationStereo" "$FOUNDATION_STEREO_URL" "$FOUNDATION_STEREO_COMMIT"

echo "Pinned external source checkouts are ready under $repo_root/external"
echo "Install each upstream environment and obtain model weights separately."
