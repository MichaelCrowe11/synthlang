# Multi-stage build for SynthLang
FROM rust:1.75 as builder

WORKDIR /app

# Copy manifests
COPY Cargo.toml Cargo.lock ./
COPY compiler/Cargo.toml ./compiler/
COPY runtime/Cargo.toml ./runtime/
COPY stdlib/Cargo.toml ./stdlib/

# Copy source code
COPY src ./src
COPY compiler ./compiler
COPY runtime ./runtime
COPY stdlib ./stdlib

# Build release binary
RUN cargo build --release

# Runtime image
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    libssl3 \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/synth /usr/local/bin/synth

# Copy examples and configs
COPY examples ./examples
COPY config ./config

# Create non-root user
RUN useradd -m -u 1001 synth && chown -R synth:synth /app
USER synth

EXPOSE 8080

ENV SYNTH_PORT=8080
ENV SYNTH_HOST=0.0.0.0

CMD ["synth", "serve", "--port", "8080"]