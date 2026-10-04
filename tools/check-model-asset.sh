#!/usr/bin/env bash
#
# Checks that the published models.zip is the file the jar will demand.
#
# Needs nothing but curl -- no gh, no login, no token. The asset is public, so
# this checks it exactly the way a biologist's Fiji will: anonymously, over
# HTTPS, following GitHub's redirect to its asset CDN.
#
# Run this AFTER publishing the model release and BEFORE publishing the jar.
# The jar hardcodes this URL and this hash, so if the check fails here, every
# install fails on first run with an error the user cannot act on.
#
#   tools/check-model-asset.sh          # reachable + right size (seconds)
#   tools/check-model-asset.sh --full   # also verify SHA-256 (downloads ~90 MB)
#
set -euo pipefail

FULL=0
[[ "${1:-}" == "--full" ]] && FULL=1

cd "$(dirname "$0")/.."
REPO="Amirhk-dev/neural-cell-classifier-fiji"
SRC="src/main/java/com/neuralimgs/fiji/ModelBootstrap.java"

# Read what the jar actually asks for, rather than taking it on trust.
BUNDLE_VERSION=$(sed -n 's/.*BUNDLE_VERSION = "\([^"]*\)".*/\1/p' "$SRC")
EXPECT_SHA=$(sed -n 's/^\t\t"\([0-9a-f]\{64\}\)";$/\1/p' "$SRC")
EXPECT_BYTES=$(sed -n 's/.*BUNDLE_BYTES = \([0-9_]*\)L;.*/\1/p' "$SRC" | tr -d _)

[[ -n "$BUNDLE_VERSION" ]] || { echo "could not read BUNDLE_VERSION from $SRC" >&2; exit 1; }
[[ -n "$EXPECT_SHA" ]] || { echo "could not read BUNDLE_SHA256 from $SRC" >&2; exit 1; }
[[ -n "$EXPECT_BYTES" ]] || { echo "could not read BUNDLE_BYTES from $SRC" >&2; exit 1; }

URL="https://github.com/$REPO/releases/download/$BUNDLE_VERSION/models.zip"

echo "the jar will request:"
echo "  url    $URL"
echo "  sha256 $EXPECT_SHA"
echo "  bytes  $EXPECT_BYTES"
echo

# -L so the redirect to the asset CDN is followed, as the plugin does.
read -r CODE SIZE < <(curl -sIL -m 60 -o /dev/null \
	-w '%{http_code} %{size_download}\n' "$URL" | tail -1)

if [[ "$CODE" != "200" ]]; then
	echo "FAIL: HTTP $CODE -- the asset is not publicly downloadable." >&2
	case "$CODE" in
		404) echo "  Either the release '$BUNDLE_VERSION' does not exist, its asset is" >&2
		     echo "  not named exactly 'models.zip', or the repository is still private." >&2 ;;
		000) echo "  No response at all -- check network access to github.com." >&2 ;;
	esac
	echo "  Publishing the jar now would ship this failure to every installer." >&2
	exit 1
fi
echo "reachable: HTTP 200"

# A HEAD through the redirect reports the real asset length; a mismatch here is
# usually "I uploaded the wrong build" and is worth catching before 90 MB of
# download proves it the slow way.
ACTUAL_BYTES=$(curl -sIL -m 60 "$URL" \
	| awk 'BEGIN{IGNORECASE=1} /^content-length:/{v=$2} END{gsub(/\r/,"",v); print v}')
if [[ -n "$ACTUAL_BYTES" && "$ACTUAL_BYTES" != "$EXPECT_BYTES" ]]; then
	echo "FAIL: published asset is $ACTUAL_BYTES bytes, jar expects $EXPECT_BYTES." >&2
	echo "  The wrong file was uploaded, or ModelBootstrap.java was not updated." >&2
	exit 1
fi
echo "size:      ${ACTUAL_BYTES:-unknown} bytes (matches)"

if [[ "$FULL" -eq 0 ]]; then
	echo
	echo "OK -- rerun with --full to verify the SHA-256 too (downloads ~90 MB)."
	exit 0
fi

echo
echo "downloading to verify SHA-256..."
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
curl -sSL -m 900 -o "$TMP/models.zip" "$URL"
ACTUAL_SHA=$(sha256sum "$TMP/models.zip" | cut -d' ' -f1)
if [[ "$ACTUAL_SHA" != "$EXPECT_SHA" ]]; then
	echo "FAIL: published asset hashes to" >&2
	echo "  $ACTUAL_SHA" >&2
	echo "but the jar will only accept" >&2
	echo "  $EXPECT_SHA" >&2
	echo "Every install would reject the download. Re-upload the right zip, or" >&2
	echo "rebuild the jar with the printed constants from prepare_model_bundle.py." >&2
	exit 1
fi
echo "sha256:    matches"
echo
echo "OK -- the published bundle is exactly what the jar expects."
