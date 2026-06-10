#!/usr/bin/env bash

set -euo pipefail

DATA_ID="1zkQ_7SfvEWRJ86dBdjmdIo5sQm87PQzO"
MODELS_ID="129Fkg_bQpVB_yN-YT5MVK6ulZS_zjhH6"
EMBEDDINGS_ID="1F0ru1XyPWco3VS2OdBAaNkR8Ismp7VAg"
RESULTS_ID="1IG99ZTXOVG78xaobXMo7DsKpokQG06-0"

usage() {
    cat <<EOF
Usage:
    $0 [data] [models] [embeddings] [results]

Example:
    $0 data models
EOF
    exit 1
}

# Check that gdown is installed
if ! command -v gdown >/dev/null 2>&1; then
    echo "Error: gdown is not installed."
    echo "Install it with: pip install gdown"
    exit 1
fi

download_item() {
    local name="$1"
    local id="$2"

    echo "Downloading '$name'..."

    if gdown "$id" -O "$name"; then
        return 0
    fi

    echo "✗ Failed to download $name"
    return 1
}

unzip_item() {
    local name="$1"

    echo "Unzipping '$name'..."
    unzip -q $name
    rm $name
}

# Require at least one argument
[[ $# -gt 0 ]] || usage

for arg in "$@"; do
    case "$arg" in
        data)
            download_item "data.zip" "$DATA_ID"
            unzip_item "data.zip"
            ;;
        models)
            download_item "models.zip" "$MODELS_ID"
            unzip_item "models.zip"
            ;;
        embeddings)
            download_item "embeddings.zip" "$EMBEDDINGS_ID"
            unzip_item "embeddings.zip"
            ;;
        results)
            download_item "results.zip" "$RESULTS_ID"
            unzip_item "results.zip"
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown argument: $arg"
            usage
            ;;
    esac
done

echo "Done!"