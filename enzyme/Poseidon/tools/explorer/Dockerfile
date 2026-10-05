ARG CE_IMAGE=ghcr.io/enzymead/compiler-explorer-builder-frontend:main
FROM ${CE_IMAGE}
ARG LLVM_VERSION=21
ARG POSEIDON_TARBALL=https://github.com/EnzymeAD/Enzyme/releases/download/nightly-poseidon/poseidon-explorer-llvm${LLVM_VERSION}.tar.gz
USER root
RUN apt-get update && apt-get install -y --no-install-recommends libmpfr6 \
 && rm -rf /var/lib/apt/lists/* \
 && mkdir -p /opt/compiler-explorer/poseidon \
 && curl -fsSL "$POSEIDON_TARBALL" | tar xz --strip-components=1 -C /opt/compiler-explorer/poseidon \
 && test -x "/usr/bin/clang++-$LLVM_VERSION" \
 && test -e "/opt/compiler-explorer/poseidon/lib/Poseidon-$LLVM_VERSION.so"
COPY config/ /app/etc/config/
COPY examples/c++/ /app/examples/c++/
COPY examples/c++/Poseidon_cancellation.cpp /app/examples/c++/default.cpp
RUN sed -i "s/@LLVM@/$LLVM_VERSION/g" /app/etc/config/c++.poseidon.properties
EXPOSE 10240
ENTRYPOINT ["node", "./out/dist/app.js", "--static", "./out/webpack/static", "--env", "poseidon", "--language", "c++"]
