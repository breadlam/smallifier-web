#!/bin/sh
# Builds the wasm module and places it next to the web app.
#   tools/build.sh            release build into web/
#   tools/build.sh dist DIR   also copy the finished site to DIR
set -e
cd "$(dirname "$0")/.."
# Two variants of the same module: without and with SVT-AV1 (the AV1 one
# is ~3x larger, so the page only fetches it when AV1 is chosen).
cargo build -p smallifier-wasm --release --target wasm32-wasip1
cp target/wasm32-wasip1/release/smallifier_wasm.wasm web/smallifier.wasm
cargo build -p smallifier-wasm --release --target wasm32-wasip1 --features av1
cp target/wasm32-wasip1/release/smallifier_wasm.wasm web/smallifier-av1.wasm
ls -l web/smallifier.wasm web/smallifier-av1.wasm
if [ "$1" = dist ]; then
  mkdir -p "$2"
  cp web/*.html web/*.js web/*.css web/smallifier.wasm web/smallifier-av1.wasm "$2"/
  echo "site in $2"
fi
