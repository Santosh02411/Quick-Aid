#!/usr/bin/env bash
# Generates a self-signed TLS certificate for LOCAL DEVELOPMENT ONLY, so
# `docker compose up` gets working HTTPS on https://localhost without any
# extra setup. Browsers will show a certificate-trust warning - that's
# expected and fine for local dev.
#
# For a real deployment, replace nginx/certs/fullchain.pem and privkey.pem
# with certificates from a real CA (e.g. Let's Encrypt via certbot) instead
# of running this script. See the README's "TLS in production" section.

set -euo pipefail

CERT_DIR="$(dirname "$0")/certs"
mkdir -p "$CERT_DIR"

if [[ -f "$CERT_DIR/fullchain.pem" && -f "$CERT_DIR/privkey.pem" ]]; then
    echo "Dev certs already exist at $CERT_DIR - remove them first if you want to regenerate."
    exit 0
fi

openssl req -x509 -nodes -newkey rsa:2048 \
    -days 365 \
    -keyout "$CERT_DIR/privkey.pem" \
    -out "$CERT_DIR/fullchain.pem" \
    -subj "/CN=localhost" \
    -addext "subjectAltName=DNS:localhost,IP:127.0.0.1"

echo "Self-signed dev certificate generated at $CERT_DIR"
echo "Your browser will warn that it's untrusted - that's expected for local dev."
