import builtins
import sys
from datetime import datetime

_original_print = builtins.print

_RED = "\033[91m"
_RESET = "\033[0m"
_ERROR_KEYWORDS = {"error", "failed", "exception", "traceback", "warning"}


def _timestamped_print(*args, **kwargs):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = " ".join(str(a) for a in args)
    msg_lower = msg.lower()

    is_error = any(kw in msg_lower for kw in _ERROR_KEYWORDS)

    file = kwargs.get("file", sys.stdout)
    is_tty = hasattr(file, "isatty") and file.isatty()
    # Also color if writing through a Tee (has .stream attr that is a tty)
    if not is_tty and hasattr(file, "stream"):
        is_tty = hasattr(file.stream, "isatty") and file.stream.isatty()

    if is_error and is_tty:
        kwargs["flush"] = kwargs.get("flush", True)
        _original_print(f"{_RED}[{timestamp}] {msg}{_RESET}", **kwargs)
    else:
        _original_print(f"[{timestamp}] {msg}", **kwargs)


builtins.print = _timestamped_print
