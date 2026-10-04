# Manual test: the model download path

`BootstrapITest` exercises `ModelBootstrap` end to end against a throwaway
localhost server, because its failure modes only ever appear on a user's very
first run and none of them are reachable from a unit test:

1. a corrupt body is rejected by the SHA-256 check, and installs nothing;
2. a real download is followed through a **302 redirect** (GitHub answers with
   one to a separate asset CDN);
3. a second call is a **no-op** -- it is pointed at a dead URL to prove it never
   touches the network;
4. a directory missing one checkpoint is **repaired**, not left half-installed.

Not under `src/test/java`: it binds a port and needs the real ~90 MB bundle, so
`mvn test` must never pick it up.

```bash
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk          # any JDK 21+
mvn -o -B -q -DskipTests compile
mkdir -p dist/manual-test-out
"$JAVA_HOME/bin/javac" -d dist/manual-test-out -cp target/classes \
    tools/manual-test/com/neuralimgs/fiji/BootstrapITest.java
"$JAVA_HOME/bin/java" -cp "target/classes:dist/manual-test-out" \
    com.neuralimgs.fiji.BootstrapITest dist/models.zip dist/manual-test-models
```

Expect it to end with `ALL CHECKS PASSED`. It needs `dist/models.zip` to exist
(see `tools/prepare_model_bundle.py`) and the SHA-256 in `ModelBootstrap.java`
to match it -- which is also a check that those two are in step.
