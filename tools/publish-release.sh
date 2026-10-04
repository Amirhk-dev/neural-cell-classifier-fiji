#!/usr/bin/env bash
#
# Publishes a release: the model bundle first, then the jar that points at it.
#
# That order is not a preference. The jar hardcodes the model bundle's URL and
# SHA-256, so a jar published before its bundle hands every installer an
# HTTP 404 on first run -- the one failure a biologist cannot diagnose. This
# script refuses to publish the jar until the bundle it names is actually
# downloadable.
#
# OPTIONAL CONVENIENCE. This is the `gh` route; `gh` is not installed on the HPC
# node, and docs/PUBLISHING.md documents the browser route that needs no CLI
# tooling at all. Both enforce the same ordering -- this one just does it
# unattended.
#
# Run from the repository root, on a machine with `gh` authenticated and a JDK
# 21+ on PATH. Safe to re-run: an already-published model bundle is left alone.
#
#   tools/publish-release.sh v1.0.0
#
set -euo pipefail

TAG="${1:-}"
if [[ -z "$TAG" ]]; then
	echo "usage: tools/publish-release.sh <jar-tag>   e.g. v1.0.0" >&2
	exit 64
fi

command -v gh >/dev/null || {
	echo "gh is not installed. Either install it (https://cli.github.com) or" >&2
	echo "follow docs/PUBLISHING.md, which publishes from the browser instead." >&2
	exit 1
}

cd "$(dirname "$0")/.."
REPO="Amirhk-dev/neural-cell-classifier-fiji"

# Both constants are read out of the source rather than passed in, so this
# script cannot publish under a version the jar does not actually request.
BUNDLE_VERSION=$(sed -n 's/.*BUNDLE_VERSION = "\([^"]*\)".*/\1/p' \
	src/main/java/com/neuralimgs/fiji/ModelBootstrap.java)
BUNDLE_SHA=$(sed -n 's/^\t\t"\([0-9a-f]\{64\}\)";$/\1/p' \
	src/main/java/com/neuralimgs/fiji/ModelBootstrap.java)
# Anchored to the artifactId: the FIRST <version> in the pom is pom-scijava's,
# not this project's, so a bare "first match" reads 43.0.0 and builds a jar name
# that does not exist.
JAR_VERSION=$(sed -n '/<artifactId>Neural_Image_Classifier<\/artifactId>/,/<\/version>/{
	s|.*<version>\(.*\)</version>.*|\1|p
}' pom.xml | head -1)

[[ -n "$BUNDLE_VERSION" ]] || { echo "could not read BUNDLE_VERSION" >&2; exit 1; }
[[ -n "$BUNDLE_SHA" ]] || { echo "could not read BUNDLE_SHA256" >&2; exit 1; }
[[ -n "$JAR_VERSION" ]] || { echo "could not read the project version" >&2; exit 1; }

echo "repo:           $REPO"
echo "jar tag:        $TAG   (pom version $JAR_VERSION)"
echo "model bundle:   $BUNDLE_VERSION"
echo "expected sha:   $BUNDLE_SHA"
echo

# -- 1. the model bundle ----------------------------------------------------

if gh release view "$BUNDLE_VERSION" --repo "$REPO" >/dev/null 2>&1; then
	echo "==> model bundle $BUNDLE_VERSION already published, leaving it alone"
else
	[[ -f dist/models.zip ]] || {
		echo "dist/models.zip not found. Build it first:" >&2
		echo "  python tools/prepare_model_bundle.py --src <checkpoint dir>" >&2
		exit 1
	}
	# The local zip must be the one the jar will demand, or every install fails
	# the integrity check instead of the download.
	LOCAL_SHA=$(sha256sum dist/models.zip | cut -d' ' -f1)
	if [[ "$LOCAL_SHA" != "$BUNDLE_SHA" ]]; then
		echo "dist/models.zip does not match ModelBootstrap.BUNDLE_SHA256:" >&2
		echo "  jar expects: $BUNDLE_SHA" >&2
		echo "  local file:  $LOCAL_SHA" >&2
		echo "Re-run prepare_model_bundle.py and paste its constants in." >&2
		exit 1
	fi
	echo "==> publishing model bundle $BUNDLE_VERSION"
	gh release create "$BUNDLE_VERSION" dist/models.zip --repo "$REPO" \
		--title "Model bundle ${BUNDLE_VERSION#models-}" \
		--notes "OPC + B3-Tub production checkpoints and thresholds.json.

Downloaded automatically by the plugin on first run -- there is nothing to do
with this file by hand. SHA-256: \`$BUNDLE_SHA\`"
fi

# -- 2. prove the jar's download will work ----------------------------------

URL="https://github.com/$REPO/releases/download/$BUNDLE_VERSION/models.zip"
echo "==> checking $URL is downloadable"
CODE=$(curl -sIL -o /dev/null -w '%{http_code}' "$URL")
if [[ "$CODE" != "200" ]]; then
	echo "models.zip is not downloadable (HTTP $CODE)." >&2
	echo "Publishing the jar now would ship a 404 to every installer." >&2
	echo "If the repository is still private, make it public first." >&2
	exit 1
fi
echo "    OK (HTTP 200)"

# -- 3. the jar -------------------------------------------------------------

JAR="target/Neural_Image_Classifier-${JAR_VERSION}.jar"
echo "==> building $JAR"
rm -rf target   # not `mvn clean`: see docs/DEVELOPING.md
mvn -B -DskipTests package
[[ -f "$JAR" ]] || { echo "expected $JAR after build" >&2; exit 1; }

echo "==> publishing $TAG"
gh release create "$TAG" "$JAR" --repo "$REPO" \
	--title "$TAG" \
	--notes "Install: download the \`.jar\` below, then in Fiji use
\`Plugins > Install...\` (or drop it into \`Fiji.app/plugins/\`) and restart.

The first run downloads its own Python environment and the trained models and
caches both -- nothing else to install. See the
[README](https://github.com/$REPO#readme).

Model bundle: \`$BUNDLE_VERSION\`."

echo
echo "done:"
echo "  https://github.com/$REPO/releases/tag/$TAG"
