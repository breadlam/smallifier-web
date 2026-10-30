// Minimal WASI preview1 for the smallifier module, with an in-memory
// filesystem (x264's two-pass stats files live here). Works in browsers,
// workers and Node.

const E = { SUCCESS: 0, BADF: 8, EXIST: 20, INVAL: 28, ISDIR: 31, NOENT: 44, NOSYS: 52, NOTDIR: 54 };
const PREOPEN_FD = 3;

class ExitError extends Error {
  constructor(code) {
    super(`wasm exited with code ${code}`);
    this.code = code;
  }
}

export class MemFS {
  constructor() {
    this.files = new Map(); // path -> { data: Uint8Array, size }
  }
  get(path) {
    return this.files.get(path);
  }
  read(path) {
    const f = this.files.get(path);
    return f ? f.data.slice(0, f.size) : null;
  }
  write(path, bytes) {
    this.files.set(path, { data: new Uint8Array(bytes), size: bytes.length });
  }
  remove(path) {
    this.files.delete(path);
  }
}

export class WASI {
  constructor({ fs = new MemFS(), stdout = console.log, stderr = console.error } = {}) {
    this.fs = fs;
    this.fds = new Map(); // fd -> { path, pos, append }
    this.nextFd = PREOPEN_FD + 1;
    this.memory = null;
    this.lines = { 1: "", 2: "" };
    this.sinks = { 1: stdout, 2: stderr };
    this.decoder = new TextDecoder();
    this.imports = { wasi_snapshot_preview1: this.#functions() };
  }

  attach(instance) {
    this.memory = instance.exports.memory;
  }

  get #view() {
    return new DataView(this.memory.buffer);
  }
  get #bytes() {
    return new Uint8Array(this.memory.buffer);
  }
  #str(ptr, len) {
    return this.decoder.decode(this.#bytes.subarray(ptr, ptr + len));
  }
  #path(dirfd, ptr, len) {
    if (dirfd !== PREOPEN_FD) return null;
    return this.#str(ptr, len).replace(/^\.?\/+/, "");
  }
  #ensure(f, size) {
    if (size <= f.data.length) return;
    const n = new Uint8Array(Math.max(size, f.data.length * 2, 4096));
    n.set(f.data.subarray(0, f.size));
    f.data = n;
  }
  #flushLine(fd, text) {
    const all = this.lines[fd] + text;
    const parts = all.split("\n");
    this.lines[fd] = parts.pop();
    for (const p of parts) this.sinks[fd](p);
  }

  #functions() {
    const self = this;
    const fns = {
      args_get: () => E.SUCCESS,
      args_sizes_get(argc, size) {
        self.#view.setUint32(argc, 0, true);
        self.#view.setUint32(size, 0, true);
        return E.SUCCESS;
      },
      environ_get: () => E.SUCCESS,
      environ_sizes_get(count, size) {
        self.#view.setUint32(count, 0, true);
        self.#view.setUint32(size, 0, true);
        return E.SUCCESS;
      },
      clock_res_get(id, out) {
        self.#view.setBigUint64(out, 1000n, true);
        return E.SUCCESS;
      },
      clock_time_get(id, precision, out) {
        const ns = id === 0 ? BigInt(Date.now()) * 1000000n : BigInt(Math.round(performance.now() * 1e6));
        self.#view.setBigUint64(out, ns, true);
        return E.SUCCESS;
      },
      random_get(buf, len) {
        crypto.getRandomValues(self.#bytes.subarray(buf, buf + len));
        return E.SUCCESS;
      },
      proc_exit(code) {
        throw new ExitError(code);
      },
      sched_yield: () => E.SUCCESS,
      fd_prestat_get(fd, buf) {
        if (fd !== PREOPEN_FD) return E.BADF;
        self.#view.setUint8(buf, 0);
        self.#view.setUint32(buf + 4, 1, true);
        return E.SUCCESS;
      },
      fd_prestat_dir_name(fd, ptr, len) {
        if (fd !== PREOPEN_FD) return E.BADF;
        self.#bytes[ptr] = 0x2f; // "/"
        return E.SUCCESS;
      },
      fd_fdstat_get(fd, buf) {
        const v = self.#view;
        let type;
        if (fd <= 2) type = 2; // character device
        else if (fd === PREOPEN_FD) type = 3; // directory
        else if (self.fds.has(fd)) type = 4; // regular file
        else return E.BADF;
        v.setUint8(buf, type);
        v.setUint16(buf + 2, 0, true);
        v.setBigUint64(buf + 8, 0xffffffffffffffffn, true);
        v.setBigUint64(buf + 16, 0xffffffffffffffffn, true);
        return E.SUCCESS;
      },
      fd_fdstat_set_flags: () => E.SUCCESS,
      fd_filestat_get(fd, buf) {
        const v = self.#view;
        for (let i = 0; i < 64; i += 8) v.setBigUint64(buf + i, 0n, true);
        if (fd <= 2) {
          v.setUint8(buf + 16, 2);
          return E.SUCCESS;
        }
        if (fd === PREOPEN_FD) {
          v.setUint8(buf + 16, 3);
          return E.SUCCESS;
        }
        const h = self.fds.get(fd);
        if (!h) return E.BADF;
        v.setUint8(buf + 16, 4);
        v.setBigUint64(buf + 24, 1n, true);
        v.setBigUint64(buf + 32, BigInt(self.fs.get(h.path)?.size ?? 0), true);
        return E.SUCCESS;
      },
      path_filestat_get(dirfd, flags, ptr, len, buf) {
        const path = self.#path(dirfd, ptr, len);
        const f = path !== null && self.fs.get(path);
        if (!f) return E.NOENT;
        const v = self.#view;
        for (let i = 0; i < 64; i += 8) v.setBigUint64(buf + i, 0n, true);
        v.setUint8(buf + 16, 4);
        v.setBigUint64(buf + 24, 1n, true);
        v.setBigUint64(buf + 32, BigInt(f.size), true);
        return E.SUCCESS;
      },
      path_open(dirfd, dirflags, ptr, len, oflags, rightsBase, rightsInh, fdflags, fdOut) {
        const path = self.#path(dirfd, ptr, len);
        if (path === null) return E.BADF;
        if (oflags & 2) return path === "" ? E.SUCCESS : E.NOTDIR;
        let f = self.fs.get(path);
        if (!f) {
          if (!(oflags & 1)) return E.NOENT;
          f = { data: new Uint8Array(4096), size: 0 };
          self.fs.files.set(path, f);
        } else if (oflags & 4) {
          return E.EXIST;
        }
        if (oflags & 8) f.size = 0;
        const fd = self.nextFd++;
        self.fds.set(fd, { path, pos: 0, append: !!(fdflags & 1) });
        self.#view.setUint32(fdOut, fd, true);
        return E.SUCCESS;
      },
      fd_close(fd) {
        if (fd <= PREOPEN_FD) return E.SUCCESS;
        return self.fds.delete(fd) ? E.SUCCESS : E.BADF;
      },
      fd_read(fd, iovs, iovsLen, nread) {
        const h = self.fds.get(fd);
        if (!h) {
          if (fd === 0) {
            self.#view.setUint32(nread, 0, true);
            return E.SUCCESS;
          }
          return E.BADF;
        }
        const f = self.fs.get(h.path);
        if (!f) return E.BADF;
        const v = self.#view;
        let total = 0;
        for (let i = 0; i < iovsLen; i++) {
          const buf = v.getUint32(iovs + 8 * i, true);
          const len = v.getUint32(iovs + 8 * i + 4, true);
          const n = Math.max(0, Math.min(len, f.size - h.pos));
          self.#bytes.set(f.data.subarray(h.pos, h.pos + n), buf);
          h.pos += n;
          total += n;
          if (n < len) break;
        }
        v.setUint32(nread, total, true);
        return E.SUCCESS;
      },
      fd_write(fd, iovs, iovsLen, nwritten) {
        const v = self.#view;
        let total = 0;
        if (fd === 1 || fd === 2) {
          let text = "";
          for (let i = 0; i < iovsLen; i++) {
            const buf = v.getUint32(iovs + 8 * i, true);
            const len = v.getUint32(iovs + 8 * i + 4, true);
            text += self.#str(buf, len);
            total += len;
          }
          self.#flushLine(fd, text);
          v.setUint32(nwritten, total, true);
          return E.SUCCESS;
        }
        const h = self.fds.get(fd);
        if (!h) return E.BADF;
        const f = self.fs.get(h.path);
        if (h.append) h.pos = f.size;
        for (let i = 0; i < iovsLen; i++) {
          const buf = v.getUint32(iovs + 8 * i, true);
          const len = v.getUint32(iovs + 8 * i + 4, true);
          self.#ensure(f, h.pos + len);
          f.data.set(self.#bytes.subarray(buf, buf + len), h.pos);
          h.pos += len;
          f.size = Math.max(f.size, h.pos);
          total += len;
        }
        v.setUint32(nwritten, total, true);
        return E.SUCCESS;
      },
      fd_seek(fd, offset, whence, out) {
        const h = self.fds.get(fd);
        if (!h) return fd <= 2 ? E.INVAL : E.BADF;
        const size = self.fs.get(h.path)?.size ?? 0;
        const base = whence === 0 ? 0 : whence === 1 ? h.pos : size;
        const pos = base + Number(offset);
        if (pos < 0) return E.INVAL;
        h.pos = pos;
        self.#view.setBigUint64(out, BigInt(pos), true);
        return E.SUCCESS;
      },
      fd_tell(fd, out) {
        const h = self.fds.get(fd);
        if (!h) return E.BADF;
        self.#view.setBigUint64(out, BigInt(h.pos), true);
        return E.SUCCESS;
      },
      path_rename(fd, oldPtr, oldLen, newFd, newPtr, newLen) {
        const from = self.#path(fd, oldPtr, oldLen);
        const to = self.#path(newFd, newPtr, newLen);
        const f = from !== null && self.fs.get(from);
        if (!f || to === null) return E.NOENT;
        self.fs.files.delete(from);
        self.fs.files.set(to, f);
        for (const h of self.fds.values()) if (h.path === from) h.path = to;
        return E.SUCCESS;
      },
      path_unlink_file(fd, ptr, len) {
        const path = self.#path(fd, ptr, len);
        return path !== null && self.fs.files.delete(path) ? E.SUCCESS : E.NOENT;
      },
    };
    // Anything else the module might import: report "not supported".
    return new Proxy(fns, { get: (t, k) => t[k] ?? (() => E.NOSYS) });
  }
}
