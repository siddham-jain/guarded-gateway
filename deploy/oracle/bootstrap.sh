#!/usr/bin/env bash
# one-time setup of an oracle cloud always free ampere (arm64) vm running ubuntu 24.04. run as root:
#   sudo GG_REPO_URL=https://github.com/<owner>/<repo>.git GG_PUBLIC_HOST=gg.example.duckdns.org \
#        GG_DEPLOY_PUBKEY="ssh-ed25519 AAAA... gg-deploy" bash bootstrap.sh
# idempotent: re-running keeps the existing .env, keys file and checkout.
set -euo pipefail

: "${GG_REPO_URL:?set GG_REPO_URL to the git url of the repo}"
: "${GG_PUBLIC_HOST:?set GG_PUBLIC_HOST to the dns name that points at this vm}"
GG_BRANCH="${GG_BRANCH:-main}"
GG_DEPLOY_PUBKEY="${GG_DEPLOY_PUBKEY:-}"
DEPLOY_USER=deploy
ROOT=/opt/gg
APP_DIR="$ROOT/app"

log() { printf '\n==> %s\n' "$*"; }

if [[ $EUID -ne 0 ]]; then
	echo "run as root (sudo)" >&2
	exit 1
fi

log "packages"
export DEBIAN_FRONTEND=noninteractive
apt-get update -q
apt-get install -y -q ca-certificates curl git iptables-persistent unattended-upgrades
dpkg-reconfigure -f noninteractive unattended-upgrades

log "docker engine and compose plugin (docker apt repo)"
if ! command -v docker >/dev/null; then
	install -m 0755 -d /etc/apt/keyrings
	curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
	chmod a+r /etc/apt/keyrings/docker.asc
	. /etc/os-release
	echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu ${VERSION_CODENAME} stable" \
		> /etc/apt/sources.list.d/docker.list
	apt-get update -q
	apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
fi
systemctl enable --now docker
# rotate container logs so a chatty gateway cannot fill the boot volume
if [[ ! -f /etc/docker/daemon.json ]]; then
	echo '{"log-driver": "json-file", "log-opts": {"max-size": "20m", "max-file": "5"}}' > /etc/docker/daemon.json
	systemctl restart docker
fi

log "firewall: oracle images reject everything but ssh in iptables; open 80 and 443"
for rule in "-p tcp --dport 80" "-p tcp --dport 443" "-p udp --dport 443"; do
	# shellcheck disable=SC2086
	iptables -C INPUT $rule -m state --state NEW -j ACCEPT 2>/dev/null \
		|| iptables -I INPUT 1 $rule -m state --state NEW -j ACCEPT
done
netfilter-persistent save

log "ssh hardening"
cat > /etc/ssh/sshd_config.d/60-gg.conf <<'EOF'
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin no
EOF
systemctl reload ssh

log "deploy user"
id "$DEPLOY_USER" >/dev/null 2>&1 || useradd --create-home --shell /bin/bash "$DEPLOY_USER"
# docker group is root-equivalent; the ci key is pinned to deploy.sh by a forced command below
usermod -aG docker "$DEPLOY_USER"
install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" -m 0755 "$ROOT" "$ROOT/bin"

log "checkout"
if [[ ! -d "$APP_DIR/.git" ]]; then
	sudo -u "$DEPLOY_USER" git clone --branch "$GG_BRANCH" "$GG_REPO_URL" "$APP_DIR"
fi
install -o root -g root -m 0755 "$APP_DIR/deploy/oracle/deploy.sh" "$ROOT/bin/deploy.sh"

log ".env and keys file"
if [[ ! -f "$APP_DIR/.env" ]]; then
	grafana_password="$(openssl rand -hex 16)"
	sed -e "s|__GG_PUBLIC_HOST__|$GG_PUBLIC_HOST|" -e "s|__GRAFANA_PASSWORD__|$grafana_password|" \
		"$APP_DIR/deploy/oracle/env.hosted.example" > "$APP_DIR/.env"
	chown "$DEPLOY_USER:$DEPLOY_USER" "$APP_DIR/.env"
	chmod 0600 "$APP_DIR/.env"
fi
if [[ ! -f "$ROOT/keys.yaml" ]]; then
	install -o "$DEPLOY_USER" -g "$DEPLOY_USER" -m 0600 "$APP_DIR/deploy/keys.hosted.example.yaml" "$ROOT/keys.yaml"
fi

if [[ -n "$GG_DEPLOY_PUBKEY" ]]; then
	log "ci deploy key (forced command)"
	auth="/home/$DEPLOY_USER/.ssh/authorized_keys"
	install -d -o "$DEPLOY_USER" -g "$DEPLOY_USER" -m 0700 "/home/$DEPLOY_USER/.ssh"
	line="command=\"$ROOT/bin/deploy.sh\",no-port-forwarding,no-agent-forwarding,no-x11-forwarding,no-pty $GG_DEPLOY_PUBKEY"
	touch "$auth"
	grep -qxF "$line" "$auth" || echo "$line" >> "$auth"
	chown "$DEPLOY_USER:$DEPLOY_USER" "$auth"
	chmod 0600 "$auth"
fi

log "build and start (first start downloads guard model weights into the gg-models volume)"
sha="$(git -C "$APP_DIR" rev-parse HEAD)"
sudo -u "$DEPLOY_USER" "$ROOT/bin/deploy.sh" deploy "$sha"

cat <<EOF

done. next:
  1. paste provider keys into $APP_DIR/.env (gemini/groq are free tiers)
  2. mint demo keys and replace the placeholders in $ROOT/keys.yaml (docs/deploy.md, step 6)
  3. sudo -u $DEPLOY_USER $ROOT/bin/deploy.sh restart
  4. curl https://$GG_PUBLIC_HOST/readyz
grafana and prometheus listen on 127.0.0.1 only: ssh -L 3000:127.0.0.1:3000 -L 9090:127.0.0.1:9090 ubuntu@<vm>
EOF
