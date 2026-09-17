#!/bin/sh
# WorkBuddy ``SessionStart`` hook — shell entry point.
#
# Why a wrapper instead of pointing the hook straight at the .py
# -------------------------------------------------------------
# WorkBuddy runs hooks under **Git Bash** on Windows (cmd.exe and PowerShell are
# explicitly unsupported). Two things are therefore *not* guaranteed at hook
# time:
#
#   1. ``python`` may not be on PATH. The hook is built to degrade silently, so
#      a missing interpreter is indistinguishable from "this store is empty" —
#      exactly the failure mode the store exists to eliminate.
#   2. The managed interpreter lives under a **versioned** directory
#      (``binaries/python/versions/3.13.12``). Hard-coding that path means the
#      hook dies the next time the runtime is upgraded, silently.
#
# So this script only answers one question — *which interpreter?* — and every
# other decision stays in ``wb_session_hook.py``. It must never exit non-zero
# and must always emit valid JSON, because the host parses stdout
# unconditionally.
#
# Environment traps this script deliberately avoids
# -------------------------------------------------
# * **No ``sort``.** Under Git Bash ``sort`` resolves to
#   ``C:\WINDOWS\system32\sort.exe``, which has no ``-V`` flag and treats the
#   argument as a *filename* (``-rV系统找不到指定的文件``). That error text then
#   lands inside ``$(...)`` and silently becomes the interpreter path. Glob
#   expansion is lexical-ordered by the shell itself, which is all we need.
# * **No ``ls``/``find``.** Same class of hijack risk; the shell's own glob is
#   both safer and faster.
# * **No ``dirname`` either.** It is an external binary inside a ``$( )``, so it
#   costs a subshell *and* a fork. Measured on this machine: swapping it for the
#   builtin ``${0%/*}`` took the sibling ``UserPromptSubmit`` wrapper from 1.20s
#   to 0.72s per invocation. Hooks are spawned per event; forks are the budget.
# * Note the corollary: a *cleaned* PATH is more reliable here than the
#   inherited one, because the inherited PATH puts Windows system directories
#   ahead of the Git Bash ``usr/bin``.

set -u

# The script's own directory, resolved to a **Windows-style** path.
#
# This must not be skipped: a POSIX path handed to a native ``python.exe`` is
# rewritten by MSYS with a *duplicated drive letter* — ``/d/repos/...`` becomes
# ``D:\d\Sync\...`` — and Python then reports the file as missing. The failure
# is silent from the host's point of view, because this hook is designed to
# degrade quietly. ``pwd -W`` is the Git Bash builtin that yields the Windows
# form; plain ``pwd`` is kept only as a fallback for shells that lack ``-W``.
# Strip the script name without forking — and handle **both** separators, because
# Windows callers produce both (the docs' example uses ``D:/...``; a path that
# was expanded from ``D:\...`` arrives with backslashes). Missing that case is
# not cosmetic: DIR silently degrades to the current directory, the script is
# then looked for in the wrong place, and the hook answers with nothing.
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
    # 1. Explicit override wins — also the escape hatch when probing goes wrong.
    if [ -n "${HGM_PYTHON:-}" ] && [ -x "${HGM_PYTHON}" ]; then
        printf '%s\n' "${HGM_PYTHON}"
        return 0
    fi

    # 2. WorkBuddy managed runtime. The glob absorbs version bumps; we keep the
    #    last match because POSIX glob expansion is lexical, so the highest
    #    version sorts last for the current naming scheme. An unmatched glob
    #    leaves ``p`` holding the literal pattern, which fails ``-x`` — safe.
    _found=""
    for p in "${HOME}"/.workbuddy/binaries/python/versions/*/python.exe; do
        if [ -x "$p" ]; then
            _found="$p"
        fi
    done
    if [ -n "$_found" ]; then
        printf '%s\n' "$_found"
        return 0
    fi

    # 3. System Python (per-user install layout).
    for p in /c/Users/*/AppData/Local/Programs/Python/Python3*/python.exe; do
        if [ -x "$p" ]; then
            printf '%s\n' "$p"
            return 0
        fi
    done

    # 4. Last resort: whatever PATH happens to offer.
    if command -v python3 >/dev/null 2>&1; then
        command -v python3
        return 0
    fi
    if command -v python >/dev/null 2>&1; then
        command -v python
        return 0
    fi

    return 1
}

PY=$(find_python) || {
    # No interpreter anywhere. Stay silent rather than block the session — a
    # memory system that bricks the editor is worse than one that is briefly
    # forgetful.
    printf '{"continue":true}\n'
    exit 0
}

# ``exec`` so the child inherits stdin (the hook payload) directly and the exit
# code propagates without an extra shell in between. Note that Git Bash
# translates the POSIX-style ``$DIR`` into a Windows path when handing it to a
# native ``python.exe``.
SCRIPT="${DIR}/wb_session_hook.py"
if [ ! -f "$SCRIPT" ]; then
    # The contract, not the filesystem: the host parses stdout unconditionally,
    # so a path problem must still produce JSON and exit 0.
    printf '{"continue":true}\n'
    exit 0
fi

exec "$PY" "$SCRIPT"
