#!/usr/bin/env python3
"""End-to-end test of the web app in a headless browser (Playwright).

  .venv/bin/python tools/browser_test.py CLIP [--target MB] [--browser chromium|firefox]
        [--preset medium|veryfast] [--fps 30|60] [--maxres auto|720|...] [--out out.mp4]

Serves web/ on localhost, loads CLIP into the page, runs it, saves the
output and checks it with ffprobe/ffmpeg.
"""
import argparse
import base64
import functools
import http.server
import json
import os
import subprocess
import sys
import threading
import time

from playwright.sync_api import sync_playwright

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "web")


def serve():
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=ROOT)
    handler.log_message = lambda *a: None
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("--target", type=float, default=10)
    ap.add_argument("--unit", default="1000000")
    ap.add_argument("--browser", default="chromium")
    ap.add_argument("--codec", default="h264")
    ap.add_argument("--preset", default=None, help="default: the codec's first preset")
    ap.add_argument("--fps", default="30")
    ap.add_argument("--maxres", default="auto")
    ap.add_argument("--audio", default="auto")
    ap.add_argument("--workers", default="auto")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--dump", default=None, help="A:B — save decoded luma of output frames A..B (diagnostics) and exit")
    ap.add_argument("--raw", action="store_true", help="with --dump: luma straight from VideoFrame.copyTo, no wasm")
    ap.add_argument("--interrupt", type=int, default=0,
                    help="cancel once this many chunks are encoded, then start again (tests resuming)")
    args = ap.parse_args()

    srv = serve()
    url = f"http://127.0.0.1:{srv.server_address[1]}/index.html"
    with sync_playwright() as p:
        browser = getattr(p, args.browser).launch()
        page = browser.new_page()
        logs = []
        page.on("console", lambda m: logs.append(f"[{m.type}] {m.text}"))
        page.on("pageerror", lambda e: logs.append(f"[pageerror] {e}"))
        page.goto(url)
        page.wait_for_function("window.__smallifierReady === true || !document.getElementById('error').hidden",
                               timeout=60000)
        if not page.is_hidden("#error"):
            print("ERROR at load:", page.text_content("#error"))
            print("\n".join(logs))
            sys.exit(1)
        page.set_input_files("#file", args.clip)
        page.wait_for_selector("#settings:not([hidden])", timeout=120000)
        if args.dump:
            a, b = map(int, args.dump.split(":"))
            b64 = page.evaluate(f"window.__smallifierDump({a}, {b}, {args.fps}, {'true' if args.raw else 'false'})")
            out = args.out or "dump.gray"
            with open(out, "wb") as f:
                f.write(base64.b64decode(b64))
            print(f"saved {out}")
            browser.close()
            return
        page.fill("#target", str(args.target))
        page.select_option("#unit", args.unit)
        page.select_option("#codec", args.codec)
        if args.preset:
            page.select_option("#preset", args.preset)
        page.select_option("#fps", args.fps)
        page.select_option("#maxres", args.maxres)
        page.select_option("#audio", args.audio)
        page.select_option("#workers", args.workers)
        print("estimate:", page.text_content("#estimate"))
        t0 = time.time()
        if args.interrupt:
            page.click("#go")
            page.wait_for_function(f"(document.getElementById('log').textContent.match(/  chunk \\d+ \\(/g) || []).length >= {args.interrupt}",
                                   timeout=args.timeout * 1000, polling=500)
            page.click("#cancel")
            page.wait_for_function("!document.getElementById('go').disabled", timeout=60000)
            print(f"cancelled after {args.interrupt} chunks ({time.time() - t0:.1f}s); reloading")
            # As after a closed tab: reload, choose the file again, and let the
            # page restore the interrupted run's settings by itself.
            page.select_option("#codec", "h264")
            page.goto(url)
            page.wait_for_function("window.__smallifierReady === true", timeout=60000)
            page.set_input_files("#file", args.clip)
            page.wait_for_selector("#resume-note:not([hidden])", timeout=60000)
            print("resume note:", page.text_content("#resume-note"))
            print("restored:", page.evaluate("['target','unit','codec','preset','fps','audio'].map(k => document.getElementById(k).value)"))
        page.click("#go")
        page.wait_for_function("window.__smallifierResult || !document.getElementById('error').hidden",
                               timeout=args.timeout * 1000, polling=1000)
        dt = time.time() - t0
        print("\n".join(l for l in logs if "[wasm]" not in l or "x264" in l))
        if not page.is_hidden("#error"):
            print("ERROR:", page.text_content("#error"))
            sys.exit(1)
        res = page.evaluate("window.__smallifierResult")
        res.pop("perChunk", None)
        print(json.dumps(res), f"wall {dt:.1f}s")
        # Can this browser actually play the result? Load, seek to the middle,
        # and confirm a frame was decoded there.
        play = page.evaluate("""() => new Promise((res) => {
            const v = document.getElementById('preview');
            const fail = () => res({ error: v.error ? v.error.code : 'unknown' });
            const seeked = () => res({ w: v.videoWidth, h: v.videoHeight, duration: v.duration, at: v.currentTime });
            const loaded = () => { v.addEventListener('seeked', seeked, { once: true }); v.currentTime = v.duration / 2; };
            v.addEventListener('error', fail, { once: true });
            if (v.readyState >= 2) loaded(); else v.addEventListener('loadeddata', loaded, { once: true });
            setTimeout(() => res({ timeout: true, readyState: v.readyState }), 20000);
        })""")
        print("in-page playback:", json.dumps(play))
        b64 = page.evaluate("""async () => {
            const blob = await (await fetch(document.getElementById('download').href)).blob();
            const buf = new Uint8Array(await blob.arrayBuffer());
            let s = ''; for (let i = 0; i < buf.length; i += 0x8000) s += String.fromCharCode(...buf.subarray(i, i + 0x8000));
            return btoa(s);
        }""")
        out = args.out or os.path.splitext(os.path.basename(args.clip))[0] + "_small.mp4"
        with open(out, "wb") as f:
            f.write(base64.b64decode(b64))
        browser.close()
    srv.shutdown()
    print(f"saved {out}: {os.path.getsize(out)} bytes")
    print(subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                          "stream=codec_name,profile,width,height,r_frame_rate,nb_frames,duration,sample_rate,channels"
                          ":format=duration,size", "-of", "compact", out], capture_output=True, text=True).stdout)
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", out, "-f", "null", "-"], capture_output=True, text=True)
    print("ffmpeg decode:", "OK" if not r.stderr.strip() else r.stderr.strip()[:2000])


if __name__ == "__main__":
    main()
