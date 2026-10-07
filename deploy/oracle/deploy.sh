#!/usr/bin/env bash
# host-side deploy entry point, installed as /opt/gg/bin/deploy.sh and used as the forced command of the ci key:
#   command="/opt/gg/bin/deploy.sh",no-port-forwarding,no-agent-forwarding,no-x11-forwarding,no-pty ssh-ed25519 AAAA... gg-deploy
# accepts exactly: deploy <40-hex sha> | restart | status
# deploy checks out the sha, rebuilds and recreates the gateway, verifies /version and /readyz, and rolls back
# to the last good sha on failure.
set -euo pipefail

APP_DIR="${GG_APP_DIR:-/opt/gg/app}"
STATE_DIR="${GG_STATE_DIR:-/opt/gg}"
LAST_GOOD="$STATE_DIR/.last_good"
VERIFY_TIMEOUT_S="${GG_VERIFY_TIMEOUT_S:-600}"

command_line="${SSH_ORIGINAL_COMMAND:-$*}"
read -r -a argv <<< "$command_line"

log() { printf '%s deploy: %s\n' "$(date -u +%FT%TZ)" "$*" >&2; }

compose() { docker compose --project-directory "$APP_DIR" "$@"; }

wait_healthy() {
	local sha="$1" deadline=$((SECONDS + VERIFY_TIMEOUT_S)) version
	while ((SECONDS < deadline)); do
		version="$(curl -fsS --max-time 3 http://127.0.0.1:8000/version 2>/dev/null || true)"
		if [[ "$version" == *"\"git_sha\":\"$sha\""* ]] && curl -fsS --max-time 3 -o /dev/null http://127.0.0.1:8000/readyz; then
			return 0
		fi
		sleep 5
	done
	return 1
}

release() {
	local sha="$1"
	git -C "$APP_DIR" fetch --quiet origin
	git -C "$APP_DIR" checkout --quiet --detach "$sha"
	GG_DEPLOY_GIT_SHA="$sha" GG_DEPLOY_BUILT_AT="$(date -u +%FT%TZ)" compose build --pull gateway
	compose up -d --remove-orphans
	wait_healthy "$sha"
}

case "${argv[0]:-}" in
deploy)
	sha="${argv[1]:-}"
	if [[ ${#argv[@]} -ne 2 || ! "$sha" =~ ^[0-9a-f]{40}$ ]]; then
		log "usage: deploy <40-hex sha>"
		exit 2
	fi
	exec 9>"$STATE_DIR/.deploy.lock"
	flock -n 9 || { log "another deploy is running"; exit 75; }
	log "deploying $sha"
	if release "$sha"; then
		echo "$sha" > "$LAST_GOOD"
		log "deployed $sha"
		exit 0
	fi
	log "verify failed for $sha"
	if [[ -s "$LAST_GOOD" ]]; then
		previous="$(cat "$LAST_GOOD")"
		log "rolling back to $previous"
		release "$previous" || log "rollback to $previous did not verify; check the vm"
	fi
	exit 1
	;;
restart)
	compose up -d --force-recreate gateway
	;;
status)
	compose ps
	curl -fsS --max-time 3 http://127.0.0.1:8000/version && echo
	;;
*)
	log "unknown command; allowed: deploy <sha> | restart | status"
	exit 2
	;;
esac
