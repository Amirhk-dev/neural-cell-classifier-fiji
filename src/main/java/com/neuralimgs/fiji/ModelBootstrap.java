package com.neuralimgs.fiji;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Puts the trained classifier checkpoints on the user's machine.
 *
 * The weights are ~45 MB per marker, which is too large to carry inside the jar
 * that gets copied into {@code Fiji.app/plugins/} and too large to keep in git.
 * They live as a single {@code models.zip} asset on a GitHub release instead,
 * and this class fetches it the first time a run needs it. A biologist
 * therefore installs one jar and nothing else: the first run downloads the
 * models the same way it already builds the Python environment, and every run
 * after that finds both cached.
 *
 * Deliberately not an unconditional download. {@link #missing} is what decides,
 * and it asks only whether the files are present -- so a user who was handed
 * the bundle on a USB stick, or who points the dialog at a directory of their
 * own retrained checkpoints, is never overwritten by a fetch they did not ask
 * for.
 */
final class ModelBootstrap {

	private ModelBootstrap() {}

	/** Reported to the user and the Log, so a stale bundle is identifiable. */
	static final String BUNDLE_VERSION = "models-v1";

	/**
	 * Release asset holding {@code thresholds.json} plus one directory per
	 * marker. Published by {@code tools/prepare_model_bundle.py}, which also
	 * prints the two constants below -- they are not editable by hand, they are
	 * copied from that script's output when a new bundle is released.
	 */
	private static final String BUNDLE_URL =
		"https://github.com/Amirhk-dev/neural-cell-classifier-fiji/releases/download/"
		+ BUNDLE_VERSION + "/models.zip";

	/**
	 * SHA-256 of that asset. A download is rejected unless it matches: the
	 * checkpoints decide every call the plugin makes, so a truncated or
	 * substituted file has to fail loudly at install time rather than quietly
	 * produce different numbers.
	 */
	private static final String BUNDLE_SHA256 =
		"0aabef9b4ebb37b7baa7859eec4a975771ce1e657884490a7a0749c9f9794335";

	/** Expected size, for the progress bar before the server reports a length. */
	private static final long BUNDLE_BYTES = 89_528_341L;

	/**
	 * System property that redirects the download to a mirror of the same asset.
	 *
	 * For a machine that cannot reach github.com -- an institute behind a proxy
	 * that blocks release CDNs -- an admin can host {@code models.zip} internally
	 * and start Fiji with {@code -Dneuralimgs.models.url=https://internal/...}.
	 * The SHA-256 check below is NOT relaxed for a mirror: the point is to change
	 * where the bytes come from, never which bytes are accepted.
	 */
	private static final String URL_PROPERTY = "neuralimgs.models.url";

	/**
	 * What a usable model directory contains, as paths relative to it.
	 *
	 * The two checkpoint directory names are not free-form: Python derives them
	 * from {@code PROD_CONFIG} via {@code prod_exp_name()} in
	 * {@code neural_imgs.inference.fixed_model_pipeline} and opens exactly those
	 * paths. Spelling them out here is what lets the plugin say "the OPC
	 * checkpoint is missing" before starting a Python worker that would take a
	 * minute to reach the same conclusion.
	 */
	private static final String[] REQUIRED = {
		"thresholds.json",
		"raw_OPC_resnet18_crop200_soft32_ch1_prod/best_model.pt",
		"raw_B3Tub_resnet18_crop200_soft16_ch3_prod/best_model.pt",
	};

	private static final int BUFFER = 1 << 16;
	/** Enough for a redirect chain (GitHub -> its asset CDN) and no more. */
	private static final int MAX_REDIRECTS = 5;

	/** Progress sink, so this class does not depend on the plugin's dialog. */
	interface Progress {
		void report(String message, long current, long maximum);
	}

	/** Required entries not present under {@code modelDir}; empty when usable. */
	static List<String> missing(final Path modelDir) {
		final List<String> absent = new ArrayList<>();
		for (final String rel : REQUIRED) {
			if (!Files.isRegularFile(modelDir.resolve(rel))) absent.add(rel);
		}
		return absent;
	}

	static boolean isInstalled(final Path modelDir) {
		return missing(modelDir).isEmpty();
	}

	/** True when this build knows where to fetch the bundle from. */
	static boolean canDownload() {
		return !BUNDLE_SHA256.startsWith("__REPLACE");
	}

	static String bundleUrl() {
		final String override = System.getProperty(URL_PROPERTY);
		return override == null || override.isBlank() ? BUNDLE_URL : override.trim();
	}

	/**
	 * Makes {@code modelDir} usable, downloading the bundle only if something is
	 * missing. A no-op on every run after the first.
	 *
	 * @return true if anything was downloaded, false if the directory was
	 *         already complete.
	 */
	static boolean ensureInstalled(final Path modelDir, final Progress progress)
		throws IOException
	{
		final List<String> absent = missing(modelDir);
		if (absent.isEmpty()) return false;

		if (!canDownload()) {
			throw new IOException(
				"This build has no model download configured, and the model "
				+ "directory is missing:\n  " + String.join("\n  ", absent)
				+ "\n\nCopy the model bundle into:\n  " + modelDir);
		}

		// Partial directories are the common case worth naming: a download
		// interrupted halfway leaves one checkpoint behind, and silently
		// re-fetching the whole bundle is the right repair -- but the user
		// should see why 90 MB is moving again.
		progress.report(absent.size() == REQUIRED.length
			? "Model bundle not found -- downloading " + BUNDLE_VERSION
			: "Model bundle incomplete (" + absent.size() + " of " + REQUIRED.length
				+ " files missing) -- re-downloading " + BUNDLE_VERSION,
			0, BUNDLE_BYTES);

		Files.createDirectories(modelDir);
		// Staged next to its destination, so the final move is within one
		// filesystem and a crash mid-download cannot leave a file that
		// `missing()` would mistake for an installed checkpoint.
		final Path staged = modelDir.resolve(".models.zip.part");
		try {
			download(bundleUrl(), staged, progress);
			verify(staged);
			progress.report("Extracting model bundle...", 0, 0);
			extract(staged, modelDir, progress);
		}
		finally {
			Files.deleteIfExists(staged);
		}

		final List<String> stillAbsent = missing(modelDir);
		if (!stillAbsent.isEmpty()) {
			throw new IOException(
				"Model bundle extracted but these files are still missing:\n  "
				+ String.join("\n  ", stillAbsent)
				+ "\n\nThe published bundle does not match what this plugin build "
				+ "expects; please report this.");
		}
		return true;
	}

	/** One-line status for the Log and the install dialog. */
	static String describe(final Path modelDir) {
		final List<String> absent = missing(modelDir);
		if (absent.isEmpty()) return "models present in " + modelDir;
		return absent.size() + " of " + REQUIRED.length + " model files missing from "
			+ modelDir;
	}

	// -- download ------------------------------------------------------------

	/**
	 * Follows redirects by hand rather than letting the client do it.
	 *
	 * A GitHub release asset answers with a redirect to a separate CDN host, and
	 * {@code HttpClient}'s NORMAL policy drops an HTTPS -> HTTP hop silently --
	 * which would surface here as an unexplained empty body rather than as a
	 * failed download. Walking the chain explicitly also bounds it.
	 */
	private static void download(
		final String url, final Path target, final Progress progress
	) throws IOException {
		final HttpClient client = HttpClient.newBuilder()
			.followRedirects(HttpClient.Redirect.NEVER)
			.connectTimeout(Duration.ofSeconds(30))
			.build();

		String current = url;
		for (int hop = 0; hop <= MAX_REDIRECTS; hop++) {
			final HttpRequest request = HttpRequest.newBuilder(URI.create(current))
				.header("Accept", "application/octet-stream")
				.GET()
				.build();

			final HttpResponse<InputStream> response;
			try {
				response = client.send(request, HttpResponse.BodyHandlers.ofInputStream());
			}
			catch (final InterruptedException e) {
				Thread.currentThread().interrupt();
				throw new IOException("Model download interrupted", e);
			}

			final int status = response.statusCode();
			if (status / 100 == 3) {
				final String location = response.headers().firstValue("location")
					.orElseThrow(() -> new IOException(
						"Model download: HTTP " + status + " without a Location header"));
				// Relative Location is legal; resolve against the hop we are on.
				current = URI.create(current).resolve(location).toString();
				try (InputStream ignored = response.body()) { /* drain + release */ }
				continue;
			}
			if (status != 200) {
				throw new IOException(
					"Model download failed: HTTP " + status + " for " + current
					+ (status == 404
						? "\n\nThe release asset for " + BUNDLE_VERSION + " is not published "
							+ "(or the repository is still private)."
						: ""));
			}

			final long total = response.headers().firstValueAsLong("content-length")
				.orElse(BUNDLE_BYTES);
			try (InputStream in = response.body();
				OutputStream out = Files.newOutputStream(target))
			{
				final byte[] buffer = new byte[BUFFER];
				long done = 0;
				long lastReported = -1;
				int n;
				while ((n = in.read(buffer)) > 0) {
					out.write(buffer, 0, n);
					done += n;
					// Repainting per 64 KB chunk would flood the Swing queue on a
					// fast link; one update per whole megabyte is still smooth.
					final long mb = done >> 20;
					if (mb != lastReported) {
						lastReported = mb;
						progress.report(
							"Downloading models: " + mb + " / " + (total >> 20) + " MB",
							done, total);
					}
				}
			}
			return;
		}
		throw new IOException("Model download: more than " + MAX_REDIRECTS + " redirects");
	}

	private static void verify(final Path zip) throws IOException {
		final String actual = sha256(zip);
		if (!actual.equalsIgnoreCase(BUNDLE_SHA256)) {
			throw new IOException(
				"Downloaded model bundle is corrupt or not the expected file.\n"
				+ "  expected SHA-256: " + BUNDLE_SHA256 + "\n"
				+ "  actual SHA-256:   " + actual + "\n\n"
				+ "Delete any partial download and try again; if it keeps failing, "
				+ "the published asset and this plugin build disagree.");
		}
	}

	private static String sha256(final Path file) throws IOException {
		final MessageDigest digest;
		try {
			digest = MessageDigest.getInstance("SHA-256");
		}
		catch (final NoSuchAlgorithmException e) {
			throw new IOException("SHA-256 unavailable in this JVM", e);
		}
		try (DigestInputStream in =
			new DigestInputStream(Files.newInputStream(file), digest))
		{
			final byte[] buffer = new byte[BUFFER];
			while (in.read(buffer) > 0) { /* digesting */ }
		}
		final StringBuilder hex = new StringBuilder(64);
		for (final byte b : digest.digest()) hex.append(String.format("%02x", b));
		return hex.toString();
	}

	private static void extract(
		final Path zip, final Path destination, final Progress progress
	) throws IOException {
		final Path root = destination.toAbsolutePath().normalize();
		try (ZipInputStream in = new ZipInputStream(Files.newInputStream(zip))) {
			ZipEntry entry;
			while ((entry = in.getNextEntry()) != null) {
				// Zip-slip: an entry named ../../something would otherwise write
				// outside the model directory. The archive is our own, but it
				// arrives over the network, so the check is not optional.
				final Path out = root.resolve(entry.getName()).normalize();
				if (!out.startsWith(root)) {
					throw new IOException(
						"Model bundle contains an entry outside the target directory: "
						+ entry.getName());
				}
				if (entry.isDirectory()) {
					Files.createDirectories(out);
					continue;
				}
				Files.createDirectories(out.getParent());
				progress.report("Extracting " + entry.getName(), 0, 0);
				// Written to a sidecar first, then moved: a checkpoint that
				// exists but is half-written is worse than one that is absent,
				// because `missing()` would call the directory complete.
				final Path temp = out.resolveSibling(out.getFileName() + ".part");
				try {
					Files.copy(in, temp, StandardCopyOption.REPLACE_EXISTING);
					Files.move(temp, out, StandardCopyOption.REPLACE_EXISTING);
				}
				finally {
					Files.deleteIfExists(temp);
				}
			}
		}
	}
}
