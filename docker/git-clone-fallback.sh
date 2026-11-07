#!/bin/sh
set -eu

repository="$1"
destination="$2"

shift 2

for base_url in "$@"; do
    rm -rf "$destination"
    if git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=60 clone \
        "${base_url}/${repository}" "$destination"; then
        exit 0
    fi
done

echo "Failed to clone ${repository} from all configured sources" >&2
exit 1
