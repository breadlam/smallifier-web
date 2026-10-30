#!/bin/sh
# Fetches the pinned C dependencies and the wasi-sdk toolchain into
# third_party/. Safe to re-run; existing checkouts are left alone.
set -e
cd "$(dirname "$0")/.."
mkdir -p third_party

X264_REV=b35605ace3ddf7c1a5d67a2eb553f034aef41d55   # stable branch, X264_BUILD 165
OPUS_TAG=v1.5.2
WASI_SDK=wasi-sdk-34
WASI_SDK_SHA256=b761e3a0721dbae9c09a0059e5fdb2bf917d1b4a8a7b430fb3b5aafb0984b2c4

if [ ! -d third_party/x264 ]; then
  git clone -q https://code.videolan.org/videolan/x264.git third_party/x264
  git -C third_party/x264 checkout -q "$X264_REV"
fi
if [ ! -d third_party/opus ]; then
  git clone -q --depth 1 -b "$OPUS_TAG" https://github.com/xiph/opus.git third_party/opus
fi
if [ ! -e third_party/wasi-sdk ]; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) arch=x86_64-linux ;;
    Linux-aarch64) arch=arm64-linux ;;
    Darwin-arm64) arch=arm64-macos ;;
    Darwin-x86_64) arch=x86_64-macos ;;
    *) echo "no wasi-sdk build for this host; set WASI_SDK_PATH" >&2; exit 1 ;;
  esac
  file="$WASI_SDK.0-$arch.tar.gz"
  curl -sSL -o "third_party/$file" "https://github.com/WebAssembly/wasi-sdk/releases/download/$WASI_SDK/$file"
  if [ "$arch" = x86_64-linux ]; then
    echo "$WASI_SDK_SHA256  third_party/$file" | sha256sum -c -
  fi
  tar -xzf "third_party/$file" -C third_party
  rm "third_party/$file"
  mv "third_party/$WASI_SDK.0-$arch" third_party/wasi-sdk
fi
echo "dependencies ready in third_party/"
