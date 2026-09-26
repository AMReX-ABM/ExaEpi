#!/usr/bin/env bash
# Make credentials available in every shell, and keep them across container rebuilds.
#
# Currently just CENSUS_API_KEY, which livelike's acs.puma() needs: as of 2026 both the ACS
# aggregate and PUMS endpoints reject keyless requests, answering HTTP 200 with an HTML
# "Missing Key" page rather than an error status, so a missing key surfaces far downstream as
# "ValueError: Expected object or value" instead of anything that names the cause.
#
# There are two ways the key can reach the container, in precedence order:
#
#   1. CENSUS_API_KEY set in the host shell before launching VS Code, forwarded by "containerEnv"
#      in devcontainer.json. This is the cleanest: nothing is stored inside the container at all.
#   2. ~/.config/exaepi/census.env, on the "exaepi-secrets" named volume. Named volumes are
#      managed by Docker on the host and are not part of the container filesystem, so this
#      survives a rebuild -- unlike the rest of /home/user, which does not.
#
# Neither is ever written into the checked-out repo.
#
# Idempotent: safe to re-run on every container create.

set -euo pipefail

SECRETS_DIR="${HOME}/.config/exaepi"
SECRETS_FILE="${SECRETS_DIR}/census.env"
BRIDGE_FILE="${HOME}/.claude/exaepi-census.env"
BASHRC="${HOME}/.bashrc"
MARKER="# exaepi: load credentials from the persistent secrets volume"

mkdir -p "${SECRETS_DIR}"
chmod 700 "${SECRETS_DIR}"

# Is the secrets directory actually a mounted volume yet, or still plain container filesystem?
# It is only a volume once devcontainer.json's mount has taken effect, which happens on the next
# rebuild after that entry was added. Compare its device against the root filesystem's: a real
# mount differs, container-local storage does not.
secrets_is_volume() {
    [ "$(stat -c %d "${SECRETS_DIR}" 2>/dev/null)" != "$(stat -c %d / 2>/dev/null)" ]
}

# One-time bridge. The secrets volume is created empty by the rebuild that first introduces it,
# so a key written before that rebuild would be lost. ~/.claude is an older named volume that
# already persists, so a copy left there carries the key across that one transition.
if [ ! -s "${SECRETS_FILE}" ] && [ -s "${BRIDGE_FILE}" ]; then
    cp "${BRIDGE_FILE}" "${SECRETS_FILE}"
    echo "setup-secrets: copied credentials from ~/.claude into ${SECRETS_DIR}"
fi

# Retire the bridge only once the destination is genuinely persistent. Deleting it while the
# secrets directory is still container-local would discard the only surviving copy on the next
# rebuild -- which is exactly the failure this guard exists to prevent.
if [ -s "${BRIDGE_FILE}" ] && [ -s "${SECRETS_FILE}" ] && secrets_is_volume; then
    rm -f "${BRIDGE_FILE}"
    echo "setup-secrets: secrets volume is live; retired the ~/.claude bridge copy"
elif [ -s "${BRIDGE_FILE}" ]; then
    echo "setup-secrets: keeping the ~/.claude bridge copy until the secrets volume is mounted"
fi

[ -f "${SECRETS_FILE}" ] && chmod 600 "${SECRETS_FILE}"

# Source it from every interactive shell, without clobbering a value forwarded from the host.
if ! grep -qF "${MARKER}" "${BASHRC}" 2>/dev/null; then
    cat >> "${BASHRC}" <<'EOF'

# exaepi: load credentials from the persistent secrets volume
if [ -z "${CENSUS_API_KEY:-}" ] && [ -r "${HOME}/.config/exaepi/census.env" ]; then
    set -a
    . "${HOME}/.config/exaepi/census.env"
    set +a
fi
EOF
    echo "setup-secrets: added credential sourcing to ~/.bashrc"
fi

if [ -s "${SECRETS_FILE}" ]; then
    echo "setup-secrets: credentials present at ${SECRETS_FILE}"
else
    echo "setup-secrets: no stored credentials yet. To add a Census API key (free, from"
    echo "               https://api.census.gov/data/key_signup.html):"
    echo "                 printf 'export CENSUS_API_KEY=%s\\n' YOURKEY > ${SECRETS_FILE}"
    echo "                 chmod 600 ${SECRETS_FILE}"
    echo "               Or export CENSUS_API_KEY on the host, which takes precedence."
fi
