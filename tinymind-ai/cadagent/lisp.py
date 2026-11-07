"""AutoLISP generation: value escaping, builders, and a syntax check.

Everything that ends up inside a generated ``.lsp`` program goes through these
helpers, so a layer name containing a quote, or a Windows path full of
backslashes, cannot break the program.

``check()`` validates parenthesis and string balance. The test suite runs it
over every generated artifact, which is the closest thing to a compiler we have
for AutoLISP outside AutoCAD.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence

# Small, reusable AutoLISP functions that recipes compose into programs.
HELPERS = {
    "split": r'''(defun tm:split (str delim / out pos)
  (setq out '())
  (while (setq pos (vl-string-search delim str))
    (setq out (cons (substr str 1 pos) out))
    (setq str (substr str (+ pos 1 (strlen delim))))
  )
  (reverse (cons str out))
)''',
    "replace-all": r'''(defun tm:replace-all (str find repl / pos)
  (setq pos (vl-string-search find str))
  (if pos
    (strcat (substr str 1 pos) repl
            (tm:replace-all (substr str (+ pos (strlen find) 1)) find repl))
    str
  )
)''',
    "csv": r'''(defun tm:csv (value)
  (tm:replace-all (tm:replace-all value "\"" "'") "," ";")
)''',
}


def lisp_string(value: object) -> str:
    """Render any value as a safe AutoLISP string literal."""
    text = str(value)
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
    text = text.replace("\t", "\\t")
    return '"%s"' % text


def path_literal(path: object) -> str:
    """AutoLISP string for a file path: forward slashes, escaped."""
    return lisp_string(str(path).replace("\\", "/"))


def number_literal(value: object) -> str:
    """Render a number the way AutoLISP expects, rejecting NaN/inf."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("not a number: %r" % (value,))

    if math.isnan(number) or math.isinf(number):
        raise ValueError("not a finite number: %r" % (value,))

    if number == int(number) and abs(number) < 1e15:
        return str(int(number))
    return ("%.6f" % number).rstrip("0").rstrip(".")


def symbol(name: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_:\-]", "-", str(name).strip()).strip("-")
    if not cleaned:
        raise ValueError("empty AutoLISP symbol")
    return cleaned


def command_symbol(name: object) -> str:
    """Turn 'hatch', 'TMK-HATCH' or 'c:tmk-hatch' into 'c:TMK-HATCH'."""
    cleaned = re.sub(r"[^A-Za-z0-9_\-:]", "-", str(name).strip())
    cleaned = cleaned.strip("-").upper()
    if cleaned.startswith("C:"):
        cleaned = cleaned[2:]
    if not cleaned:
        raise ValueError("empty command name")
    if not cleaned.startswith("TMK-"):
        cleaned = "TMK-" + cleaned
    return "c:" + cleaned


def quoted_pairs(pairs: Iterable) -> str:
    """A quoted list of (string number) pairs, e.g. layer names and colours."""
    rows = []
    for first, second in pairs:
        rows.append("  (%s %s)" % (lisp_string(first), number_literal(second)))
    if not rows:
        return "(list)"
    return "(list\n%s\n)" % "\n".join(rows)


def _args_text(params: Sequence[str], locals_: Sequence[str]) -> str:
    params = [symbol(name) for name in params]
    locals_ = [symbol(name) for name in locals_]

    if params and locals_:
        return "%s / %s" % (" ".join(params), " ".join(locals_))
    if locals_:
        return " / %s" % " ".join(locals_)
    return " ".join(params)


def defun(
    name: object,
    params: Sequence[str] = (),
    locals_: Sequence[str] = (),
    body: Sequence[str] = (),
    doc: str = None,
    command: bool = False,
    epilogue: Sequence[str] = None,
) -> str:
    """Build a balanced ``(defun ...)`` form.

    ``command=True`` names it ``c:TMK-...`` and appends the conventional silent
    ``(princ)``, so running the command does not echo a return value.
    """
    symbol_name = command_symbol(name) if command else symbol(name)
    lines = ["(defun %s (%s)" % (symbol_name, _args_text(params, locals_))]

    if doc:
        for line in str(doc).splitlines():
            lines.append("  ;; " + line.strip())

    for entry in body:
        for line in str(entry).splitlines():
            lines.append(("  " + line).rstrip() if line.strip() else "")

    if epilogue is None:
        epilogue = ("(princ)",) if command else ()
    for line in epilogue:
        lines.append("  " + line)

    lines.append(")")
    return "\n".join(lines)


def program(
    title: str,
    forms: Sequence[str],
    helpers: Sequence[str] = (),
    command: str = None,
) -> str:
    """Assemble a loadable .lsp file: header, shared helpers, forms."""
    header = [";; amu: " + title]
    if command:
        header.append(";; APPLOAD this file, then run: " + command.lstrip("c:").upper())
    header.append("(vl-load-com)")

    parts = ["\n".join(header)]
    for name in helpers:
        try:
            parts.append(HELPERS[name])
        except KeyError:
            raise KeyError("unknown AutoLISP helper %r" % (name,))
    parts.extend(forms)

    return "\n\n".join(part.strip("\n") for part in parts) + "\n"


def check(text: str) -> List[str]:
    """Return a list of structural problems, ignoring comments and strings."""
    problems: List[str] = []
    depth = 0
    in_string = False
    escaped = False
    in_comment = False
    string_start = 0
    line = 1

    for char in text:
        if char == "\n":
            line += 1
            in_comment = False
            escaped = False
            continue

        if in_comment:
            continue

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == ";":
            in_comment = True
        elif char == '"':
            in_string = True
            string_start = line
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                problems.append("line %d: closing parenthesis with no opener" % line)
                depth = 0

    if in_string:
        problems.append("line %d: unterminated string" % string_start)
    if depth:
        problems.append("unbalanced parentheses: %d unclosed" % depth)

    return problems
