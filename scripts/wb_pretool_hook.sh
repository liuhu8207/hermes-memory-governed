#!/bin/sh
# WorkBuddy ``PreToolUse`` hook — shell entry point.
#
# Same three rules as its siblings (see wb_prompt_hook.sh for the measurements):
#
#   * no ``dirname`` — it is an external binary inside ``$( )``, and every fork
#     is paid per hook invocation;
#   * both path separators handled, because Windows callers produce both;
#   * the host parses stdout unconditionally, so this must always print valid
#     JSON and exit 0.
#
# The extra rule here is about *blast radius*: this hook can refuse an edit, so a
# bug in it could stop all work. The interpreter check below therefore leans the
# other way — if anything at all is wrong, it prints ``{"continue": true}`` and
# lets the edit through. An unenforced rule is recoverable; a session that cannot
# edit a file is not.

set -u

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

SCRIPT="${DIR}/wb_pretool_hook.py"
PY=$(find_python) || {
    printf '{"continue":true}\n'
    exit 0
}
if [ ! -f "$SCRIPT" ]; then
    printf '{"continue":true}\n'
    exit 0
fi

exec "$PY" "$SCRIPT"
