#!/bin/sh
# WorkBuddy ``UserPromptSubmit`` hook — shell entry point.
#
# Why this is cheap, and why that matters
# --------------------------------------
# This hook runs before **every** prompt. Measured 2026-09-17 on this machine:
#
#     sh + python start .............. 0.71s
#     recall --lexical-only .......... 2.16s
#     recall (semantic) .............. 2.83s
#
# So this path must not touch the CLI's recall commands. It does not need to:
# matching is done in-process against a JSON snapshot, and the only import it
# makes is ``memory_cli`` for its tokenizer — whose own top-level imports are
# stdlib only (LanceDB is loaded lazily, per command). Any Python 3 works here,
# which is also why interpreter discovery below stays simple.
#
# The two rules the session hook learned the hard way, kept here:
#
#   * ``sort`` / ``ls`` / ``find`` are hijacked by Windows system32 under Git
#     Bash. ``sort -rV`` there reads its argument as a *filename* and prints
#     "系统找不到指定的文件" — which then lands inside ``$(...)`` and silently
#     becomes the interpreter path. The shell's own glob is safer, and its
#     lexical order already puts the highest version last.
#   * A POSIX path handed to a native ``python.exe`` gets a duplicated drive
#     letter from MSYS (``/d/repos`` -> ``D:\d\Sync``), so the directory must be
#     converted with ``pwd -W`` first.
#
# It must never exit non-zero and must always print valid JSON: the host parses
# stdout unconditionally.

set -u

# Strip the script name without forking.
#
# ``dirname`` is an external binary inside a command substitution: a subshell
# *and* a fork. Measured on this machine, replacing it with the builtin took
# this wrapper from 1.20s to 0.72s per invocation — and this one runs before
# every prompt. Forks are the budget.
#
# Both separators are handled because Windows callers produce both: the docs
# example uses ``D:/...`` while a test or a shell that expands ``D:\...`` gives
# backslashes. Missing that case is not cosmetic — DIR silently degrades to the
# current directory, the script is looked for in the wrong place, and the hook
# answers with nothing.
case "$0" in
    */*)  DIR=${0%/*} ;;
    *\\*) DIR=${0%\\*} ;;
    *)    DIR="." ;;
esac
case "$DIR" in
    [A-Za-z]:[\\/]*) ;;                     # already Windows-absolute
    *) DIR=$(cd -- "$DIR" 2>/dev/null && (pwd -W 2>/dev/null || pwd)) || DIR="." ;;
esac

find_python() {
    if [ -n "${HGM_PYTHON:-}" ] && [ -x "${HGM_PYTHON}" ]; then
        printf '%s\n' "${HGM_PYTHON}"
        return 0
    fi
    _found=""
    for p in "${HOME}"/.workbuddy/binaries/python/versions/*/python.exe; do
        [ -x "$p" ] && _found="$p"
    done
    if [ -n "$_found" ]; then
        printf '%s\n' "$_found"
        return 0
    fi
    for p in /c/Users/*/AppData/Local/Programs/Python/Python3*/python.exe; do
        [ -x "$p" ] && { printf '%s\n' "$p"; return 0; }
    done
    command -v python3 2>/dev/null && return 0
    command -v python  2>/dev/null && return 0
    return 1
}

PY=$(find_python) || {
    # No interpreter: stay silent rather than stall the conversation. The agent
    # simply does not get the convenience; it can still ask the CLI itself.
    printf '{"continue":true}\n'
    exit 0
}

SCRIPT="${DIR}/wb_prompt_hook.py"
if [ ! -f "$SCRIPT" ]; then
    # Guards the contract, not just the filesystem: whatever went wrong with the
    # path, the host still gets parseable JSON and a zero exit. A hook that
    # answers with nothing is invisible; one that answers with a traceback is a
    # session error.
    printf '{"continue":true}\n'
    exit 0
fi

exec "$PY" "$SCRIPT"
