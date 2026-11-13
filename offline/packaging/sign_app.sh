#!/bin/bash
# Sign HoRUS.app from the inside out.
#
#   bash sign_app.sh dist/HoRUS.app            # ad hoc (runs on this Mac; Gatekeeper warns elsewhere)
#   bash sign_app.sh dist/HoRUS.app "Developer ID Application: Name (TEAMID)"
#
# Every Mach-O file (Python extensions, Qt, onnxruntime, CTranslate2, and the llama.cpp server that
# ships as data) is signed first, then each framework, then the bundle. Apple silicon refuses to run
# unsigned code, and notarisation needs every binary signed with the hardened runtime.
set -euo pipefail

APP="${1:?usage: sign_app.sh <HoRUS.app> [identity]}"
ID="${2:--}"
HERE="$(cd "$(dirname "$0")" && pwd)"
ENT="$HERE/entitlements.plist"

opts=(--force --sign "$ID")
if [ "$ID" != "-" ]; then
  opts+=(--options runtime --timestamp)
fi

# deepest paths first, so a container is signed after what it contains
by_depth() { awk '{ print gsub("/", "/"), $0 }' | sort -rn | cut -d' ' -f2-; }

n=0
while IFS= read -r f; do
  codesign "${opts[@]}" "$f"
  n=$((n + 1))
done < <(find "$APP/Contents" -type f ! -path "*/_CodeSignature/*" ! -name "*.gguf" ! -name "*.bin" \
           ! -name "*.json" ! -name "*.npz" ! -name "*.csv" ! -name "*.gz" ! -name "*.pkl" ! -name "*.onnx" \
           -exec sh -c 'file -b "$1" | grep -q "Mach-O" && echo "$1"' _ {} \; | by_depth)

while IFS= read -r fw; do
  codesign "${opts[@]}" "$fw"
done < <(find "$APP/Contents" -type d -name "*.framework" | by_depth)

# signing rewrote the llama.cpp binaries: update their checksums before the bundle seal covers them
python3 "$HERE/refresh_manifest.py" "$APP"

codesign "${opts[@]}" --entitlements "$ENT" "$APP"
codesign --verify --deep --strict "$APP"
echo "Signed $n binaries and $APP ($([ "$ID" = "-" ] && echo "ad hoc" || echo "$ID"))"
