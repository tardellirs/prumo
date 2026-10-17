"""Run Python that comes inside a request in a separate, isolated process: no network, no writes outside a fresh
temporary directory, no subprocesses, and a deterministic execution budget.

Layers (each one alone is enough to stop the common cases; together they are defence in depth):

1. OS sandbox when the host has one: `sandbox-exec` on macOS (deny network, deny writes outside the temp dir);
   on Linux `bwrap` (all namespaces unshared, read-only root, private /tmp) or else `unshare -rn` (no network).
2. A fresh `python -I -S -B` child in an empty temporary directory, minimal environment, PYTHONHASHSEED=0.
3. Resource limits in the child: CPU seconds, address space, file size, no new processes.
4. A CPython audit hook installed before the code runs: refuses sockets, subprocesses, exec/fork/spawn, ctypes,
   signals to other processes, file writes outside the temp dir and file reads outside the Python installation.
5. A deterministic **line budget** (`sys.settrace` counts executed lines and calls; past LINES the run is stopped).
   The answer therefore never depends on the speed of the machine. A wall-clock timeout remains only as a safety
   net for C-level loops that the line counter cannot see; a run that hits it is treated as "cannot tell".

Nothing from the request ever runs in the engine's own process.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

LINES = 1_000_000          # executed lines + calls per run (deterministic budget)
WALL = 1.0                 # seconds; safety net only (C-level loops the line budget cannot see)
CPU = 1                    # seconds of CPU (RLIMIT_CPU), safety net only

CHILD = r'''
import sys, os, json, ast, io, contextlib, random
req = json.loads(sys.stdin.read())
TMP = os.path.realpath(os.getcwd())
READ_OK = tuple(os.path.realpath(p) for p in {sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix, TMP}) + ("/dev/null", "/dev/urandom")
try:
    import resource
    for lim, val in ((resource.RLIMIT_CPU, req["cpu"]), (resource.RLIMIT_FSIZE, 1 << 20), (resource.RLIMIT_AS, 2 << 30), (resource.RLIMIT_NPROC, 0)):
        try:
            resource.setrlimit(lim, (val, val))
        except (ValueError, OSError):
            pass
except ImportError:
    pass
BLOCK_PREFIX = ("socket.", "subprocess.", "os.system", "os.exec", "os.posix_spawn", "os.spawn", "os.fork", "os.forkpty",
                "os.kill", "os.killpg", "os.setuid", "os.putenv", "os.unsetenv", "ctypes.", "pty.", "webbrowser.",
                "urllib.Request", "http.client.", "ftplib.", "smtplib.", "imaplib.", "poplib.", "nntplib.", "telnetlib.",
                "sys.settrace", "sys.setprofile", "sys.addaudithook", "sys.remote_exec", "_posixsubprocess.", "signal.",
                "os.chdir", "os.chroot", "mmap.", "syslog.", "resource.setrlimit", "os.system")
WRITE_EVENTS = ("os.remove", "os.unlink", "os.rename", "os.replace", "os.rmdir", "os.mkdir", "os.makedirs", "os.chmod", "os.chown",
                "os.link", "os.symlink", "os.truncate", "os.utime", "shutil.rmtree", "shutil.copyfile", "shutil.copytree",
                "shutil.move", "shutil.chown", "os.chflags", "os.lchmod", "os.mkfifo", "os.mknod")
BLOCK_IMPORT = {"ctypes", "_ctypes", "socket", "_socket", "ssl", "_ssl", "subprocess", "_posixsubprocess", "multiprocessing",
                "_multiprocessing", "mmap", "pty", "fcntl", "resource", "signal", "_signal", "webbrowser", "urllib.request",
                "http.client", "ftplib", "smtplib", "asyncio", "concurrent", "threading", "_thread"}
ARMED = [False]
def inside(p, roots):
    try:
        rp = os.path.realpath(os.fsdecode(p))
    except Exception:
        return False
    return any(rp == r or rp.startswith(r.rstrip("/") + "/") for r in roots)
def hook(event, args):
    if not ARMED[0] or (event == "sys.settrace" and COUNT[0] > LIMIT):
        return
    if event.startswith(BLOCK_PREFIX):
        raise PermissionError("sandbox: " + event)
    if event == "import" and args and (args[0] in BLOCK_IMPORT or args[0].split(".")[0] in BLOCK_IMPORT):
        raise ImportError("sandbox: import " + str(args[0]))
    if event == "open" and args:
        path, mode, flags = (list(args) + [None, None, None])[:3]
        if isinstance(path, int) or path is None:
            return
        writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC))
        if writing and not inside(path, (TMP,)):
            raise PermissionError("sandbox: write outside temp dir")
        if not writing and not inside(path, READ_OK):
            raise PermissionError("sandbox: read outside python/temp")
    if event in WRITE_EVENTS:
        for a in args[:2]:
            if isinstance(a, (str, bytes, os.PathLike)) and not inside(a, (TMP,)):
                raise PermissionError("sandbox: " + event)
sys.addaudithook(hook)
LIMIT = int(req["lines"])
COUNT = [0]
class Budget(BaseException):
    pass
def tracer(frame, event, arg):
    COUNT[0] += 1
    if COUNT[0] > LIMIT:
        raise Budget()
    return tracer
def lit(s):
    try:
        return True, ast.literal_eval(s)
    except Exception:
        return False, None
out = {"ok": False}
random.seed(0)
ns = {"__name__": "__main__", "__builtins__": __builtins__}
buf = io.StringIO()
try:
    code = compile(req["code"], "<code>", "exec")
    call = compile(req["call"], "<call>", "eval") if req.get("call") else None
    tests = compile(req["tests"], "<tests>", "exec") if req.get("tests") else None
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        sys.settrace(tracer)
        ARMED[0] = True
        try:
            exec(code, ns)
            if req["mode"] == "output":
                val = eval(call, ns)
            else:
                exec(tests, ns)
                if callable(ns.get("check")) and req.get("entry") in ns:
                    ns["check"](ns[req["entry"]])
        finally:
            ARMED[0] = False
            sys.settrace(None)
    if req["mode"] == "output":
        m = []
        for c in req["candidates"]:
            ok, v = lit(c)
            same = False
            if ok:
                try:
                    same = (type(v) == type(val) and bool(v == val)) or repr(v) == repr(val)
                except Exception:
                    same = False
            if not same:
                same = c.strip() == repr(val) or c.strip() == str(val)
            m.append(bool(same))
        out.update(ok=True, matches=m)
    else:
        out.update(ok=True, passes=True)
except Budget:
    out.update(ok=False, error="budget")
except AssertionError:
    out.update(ok=True, passes=False, error="AssertionError") if req["mode"] == "tests" else out.update(error="AssertionError")
except BaseException as e:
    msg = type(e).__name__ + ": " + str(e)[:200]
    out.update(error=msg)
    if req["mode"] == "tests" and not msg.startswith(("PermissionError: sandbox", "ImportError: sandbox", "MemoryError", "RecursionError")):
        out.update(ok=True, passes=False)
ARMED[0] = False
sys.__stdout__.write("\n__RESULT__" + json.dumps(out))
sys.__stdout__.flush()
'''


def _probe(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, timeout=5).returncode == 0
    except Exception:
        return False


def _os_wrapper(tmp):
    """The OS-level sandbox prefix for this host (probed once)."""
    global _WRAP
    if _WRAP is None:
        if sys.platform == "darwin" and shutil.which("sandbox-exec") and _probe(["sandbox-exec", "-p", "(version 1)(allow default)", "/usr/bin/true"]):
            _WRAP = "sandbox-exec"
        elif sys.platform.startswith("linux") and shutil.which("bwrap") and _probe(["bwrap", "--ro-bind", "/", "/", "--unshare-all", "true"]):
            _WRAP = "bwrap"
        elif sys.platform.startswith("linux") and shutil.which("unshare") and _probe(["unshare", "-rn", "true"]):
            _WRAP = "unshare"
        else:
            _WRAP = "none"
    if _WRAP == "sandbox-exec":
        prof = ('(version 1)(allow default)(deny network*)(deny file-write*)'
                f'(allow file-write* (subpath "{tmp}") (literal "/dev/null"))')
        return ["sandbox-exec", "-p", prof]
    if _WRAP == "bwrap":
        return ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--tmpfs", "/tmp", "--bind", tmp, tmp, "--chdir", tmp,
                "--unshare-all", "--die-with-parent", "--new-session"]
    if _WRAP == "unshare":
        return ["unshare", "-rn"]
    return []


_WRAP = None


def isolation():
    _os_wrapper("/tmp")
    return _WRAP


def run(req, lines=LINES, wall=WALL):
    """req: {"mode": "output", "code", "call", "candidates"} or {"mode": "tests", "code", "tests", "entry"}."""
    req = dict(req, lines=lines, cpu=CPU)
    with tempfile.TemporaryDirectory(prefix="zr-") as d:
        d = os.path.realpath(d)
        cmd = _os_wrapper(d) + [sys.executable, "-I", "-S", "-B", "-c", CHILD]
        try:
            p = subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True, timeout=wall, cwd=d,
                               env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "LC_ALL": "C.UTF-8", "TMPDIR": d})
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout"}
        tail = p.stdout.rsplit("\n__RESULT__", 1)
        if len(tail) != 2:
            return {"ok": False, "error": "no result: " + p.stderr[-300:]}
        try:
            return json.loads(tail[1])
        except Exception:
            return {"ok": False, "error": "bad result"}
