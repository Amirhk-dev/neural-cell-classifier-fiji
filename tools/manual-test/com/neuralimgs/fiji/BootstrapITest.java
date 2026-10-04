package com.neuralimgs.fiji;

import com.sun.net.httpserver.HttpServer;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.file.*;
import java.util.List;

/** Throwaway harness: exercises the real install path over localhost. */
public class BootstrapITest {
	public static void main(String[] args) throws Exception {
		Path zip = Paths.get(args[0]);
		Path dest = Paths.get(args[1]);
		byte[] body = Files.readAllBytes(zip);

		HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
		int port = server.getAddress().getPort();
		// Hop 1: a 302 to a different path, mimicking GitHub -> asset CDN.
		server.createContext("/release", ex -> {
			ex.getResponseHeaders().add("Location", "/cdn/models.zip");
			ex.sendResponseHeaders(302, -1);
			ex.close();
		});
		server.createContext("/cdn/models.zip", ex -> {
			ex.sendResponseHeaders(200, body.length);
			try (OutputStream os = ex.getResponseBody()) { os.write(body); }
		});
		// Hop that serves a deliberately corrupt body, for the SHA check.
		server.createContext("/corrupt", ex -> {
			byte[] bad = body.clone();
			bad[bad.length / 2] ^= 0xFF;
			ex.sendResponseHeaders(200, bad.length);
			try (OutputStream os = ex.getResponseBody()) { os.write(bad); }
		});
		server.setExecutor(null);
		server.start();

		ModelBootstrap.Progress p = (m, c, max) -> {
			if (c == 0 || c == max) System.out.println("  [progress] " + m);
		};

		System.out.println("missing on empty dir: " + ModelBootstrap.missing(dest).size());
		System.out.println("describe: " + ModelBootstrap.describe(dest));

		// 1. corrupt download must be rejected and must not install anything
		System.setProperty("neuralimgs.models.url", "http://127.0.0.1:" + port + "/corrupt");
		try {
			ModelBootstrap.ensureInstalled(dest, p);
			throw new AssertionError("FAIL: corrupt bundle was accepted");
		} catch (java.io.IOException e) {
			System.out.println("corrupt rejected: " + e.getMessage().split("\n")[0]);
		}
		if (ModelBootstrap.isInstalled(dest)) throw new AssertionError("FAIL: installed from corrupt zip");
		if (Files.exists(dest.resolve(".models.zip.part"))) throw new AssertionError("FAIL: staging file left behind");

		// 2. real download through a redirect
		System.setProperty("neuralimgs.models.url", "http://127.0.0.1:" + port + "/release");
		boolean did = ModelBootstrap.ensureInstalled(dest, p);
		System.out.println("downloaded: " + did + ", installed: " + ModelBootstrap.isInstalled(dest));
		System.out.println("describe: " + ModelBootstrap.describe(dest));
		for (Path f : (Iterable<Path>) Files.walk(dest).filter(Files::isRegularFile).sorted()::iterator) {
			System.out.println("  " + dest.relativize(f) + "  " + Files.size(f) + " bytes");
		}

		// 3. second call is a no-op (no network needed): point at a dead URL
		System.setProperty("neuralimgs.models.url", "http://127.0.0.1:1/nope");
		System.out.println("second call downloaded: " + ModelBootstrap.ensureInstalled(dest, p));

		// 4. a partial directory is repaired
		Files.delete(dest.resolve("raw_OPC_resnet18_crop200_soft32_ch1_prod/best_model.pt"));
		List<String> gone = ModelBootstrap.missing(dest);
		System.out.println("after deleting one checkpoint, missing: " + gone);
		System.setProperty("neuralimgs.models.url", "http://127.0.0.1:" + port + "/release");
		System.out.println("repair downloaded: " + ModelBootstrap.ensureInstalled(dest, p)
			+ ", installed: " + ModelBootstrap.isInstalled(dest));

		server.stop(0);
		System.out.println("ALL CHECKS PASSED");
		System.exit(0);
	}
}
