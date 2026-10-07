# Deploying GG

GG runs as one container (`gateway`) next to Redis, Prometheus and Grafana from `docker-compose.yml`. The hosted
demo runs that compose stack on an Oracle Cloud Always Free Ampere VM behind Caddy (automatic TLS). Cloud Run is
the documented fallback.

## Image

`Dockerfile` builds a multi-stage, uv-based `python:3.13-slim` image (~0.9 GB uncompressed on arm64) that runs as
uid 10001 with a read-only root filesystem in compose.

- The `ml` extra (onnxruntime, fastembed, presidio + spaCy `en_core_web_sm`) is installed; torch is not. Build
  with `--build-arg EXTRAS=""` for a slim image whose model-backed guards and semantic cache are off.
- tiktoken's `o200k_base` is cached at build time (`TIKTOKEN_CACHE_DIR=/opt/gg/tiktoken`).
- Guard model weights are **not** in the image. `GG_MODELS_DIR=/opt/gg/models` is the `gg-models` volume; on first
  start the gateway fetches the ungated artefacts at their pinned revisions (`gg.guardrails.ml.artefacts`) and the
  bge-small embedder into it (~350 MB, about 2-3 minutes), then reuses them on every restart. `/readyz` stays 503
  until they are loaded. Locally exported artefacts (e.g. `granite-guardian-hap-38m-int8`) can be copied in:
  `docker run --rm -v gg_gg-models:/m -v "$PWD/.models:/src:ro" alpine sh -c "cp -a /src/. /m/ && chown -R 10001:10001 /m"`.
- Defaults: `GG_ENV=prod`, `GG_LOG_FORMAT=json`, bind `0.0.0.0:8000`, healthcheck on `/healthz`.
  `FORWARDED_ALLOW_IPS` trusts private ranges, so the client ip behind Caddy is the real one.
- Build args `GIT_SHA` and `BUILT_AT` surface in `/version`.

## Local

```sh
cp .env.example .env          # set GG_GRAFANA_ADMIN_PASSWORD; provider keys optional
docker compose up -d --build  # gateway :8000, redis, prometheus :9090, grafana :3000 (all on 127.0.0.1)
curl localhost:8000/readyz
```

Without provider keys use the mock profile: `GG_MODEL_PROFILE=ci` and `GG_ENV=test` in `.env`, then call the gateway
with the fixture key from `tests/fixtures/keys/tokens.yaml` after setting
`GG_DEPLOY_KEYS_FILE=./tests/fixtures/keys/keys.yaml`.

Compose-only variables (`GG_DEPLOY_*`, read by compose, ignored by the gateway):

| Variable | Default | Meaning |
|---|---|---|
| `GG_DEPLOY_PORT` | `8000` | host port (always bound to 127.0.0.1) |
| `GG_DEPLOY_KEYS_FILE` | `./config/keys.yaml` | keys file mounted at `/run/gg/keys.yaml` |
| `GG_DEPLOY_LANGFUSE_HOST` | `http://langfuse-web:3000` | Langfuse url as seen from the container |
| `GG_DEPLOY_GIT_SHA`, `GG_DEPLOY_BUILT_AT` | empty | build args for `/version` |
| `GG_PUBLIC_HOST` | `localhost` | dns name Caddy serves (profile `hosted`) |

## Demo safety

| Control | How |
|---|---|
| Kill switch | `GG_MAINTENANCE=1` in `.env`, then `deploy.sh restart`: `/v1/*` answers 503 `maintenance` for every key not tagged `admin`; `/healthz`, `/readyz`, `/version` stay up. Faster still: `docker compose stop caddy`. |
| Failed-auth limiter | `GG_AUTH_FAILURES__MAX_FAILURES` (10) 401s from one ip within `GG_AUTH_FAILURES__WINDOW_S` (60 s) earn that ip a 429 with `retry-after` on `/v1/*` until the window ends. Counted in Redis when configured (shared across restarts), in memory otherwise. |
| Demo keys | `deploy/keys.hosted.example.yaml`: graders get `gg/auto`, `gg/weak`, `gg/strong`, 512 completion tokens, 10 rpm, $0.25/month; presenter may call `gg/demo-chaos`; `ops` is tagged `admin`. Sum of all budgets ≤ $3. |
| Spend caps | `GG_SPEND_CAP_*` per provider, fail closed. |
| Public surface | Caddy proxies only `/v1/*`, `/healthz`, `/readyz`, `/version`; body ≤ 256 KB. Redis, Prometheus, Grafana and Langfuse bind to 127.0.0.1. |
| Chaos demo | `GG_MODEL_PROFILE=demo`: `gg/demo-chaos` = `mock/down` (always 529) → Gemini Flash-Lite → `mock/echo`. |

## Hosted: Oracle Cloud Always Free (primary)

1. **Account.** Sign up at cloud.oracle.com (card needed for verification). Pick the home region carefully; it
   cannot change. Optionally upgrade to Pay As You Go (still $0 inside Always Free) with a $1 budget alert so idle
   instances are not reclaimed.
2. **Network.** Networking → Virtual cloud networks → "Start VCN wizard" (internet connectivity). In the public
   subnet's security list add ingress TCP 80 and 443 from `0.0.0.0/0` (and UDP 443 for HTTP/3); restrict TCP 22 to
   your ip.
3. **Instance.** Compute → Create instance: image Canonical Ubuntu 24.04 (aarch64), shape `VM.Standard.A1.Flex`
   4 OCPU / 24 GB, boot volume 100 GB, your SSH public key. On "out of capacity", retry off-peak or try another
   availability domain. Optional: paste `deploy/oracle/cloud-init.yaml` (with its three values filled) into the
   cloud-init box and skip step 5.
4. **DNS.** Point a name at the instance's public ip, e.g. a free DuckDNS subdomain (`gg-demo.duckdns.org`).
5. **Bootstrap.** Generate a deploy key pair on your machine
   (`ssh-keygen -t ed25519 -f gg-deploy -N "" -C gg-deploy`), then on the VM:

   ```sh
   git clone https://github.com/<owner>/<repo>.git /tmp/gg && cd /tmp/gg
   sudo GG_REPO_URL=https://github.com/<owner>/<repo>.git GG_PUBLIC_HOST=gg-demo.duckdns.org \
        GG_DEPLOY_PUBKEY="$(cat gg-deploy.pub)" bash deploy/oracle/bootstrap.sh
   ```

   It installs Docker Engine + compose from Docker's apt repo, opens 80/443 in Oracle's iptables rules
   (`netfilter-persistent save`), disables SSH passwords, enables unattended upgrades, creates the `deploy` user,
   clones to `/opt/gg/app`, writes `/opt/gg/app/.env` from `deploy/oracle/env.hosted.example` (0600, random
   Grafana password, `COMPOSE_PROFILES=hosted`, `GG_MODEL_PROFILE=demo`), copies the keys template to
   `/opt/gg/keys.yaml`, installs `/opt/gg/bin/deploy.sh` as the forced command of the deploy key, then builds and
   starts the stack.
6. **Secrets on the VM.** `sudo -u deploy nano /opt/gg/app/.env`: provider keys (Gemini and Groq free tiers first),
   optional Langfuse Cloud keys. Mint each demo key and swap it into `/opt/gg/keys.yaml`:

   ```sh
   cd /opt/gg/app && sudo -u deploy docker compose run --rm --no-deps gateway gg keys create --id grader-1 --name "Grader 1"
   ```

   The token is printed once; hand it to its holder privately. Then `sudo -u deploy /opt/gg/bin/deploy.sh restart`.
7. **Check.** `curl https://gg-demo.duckdns.org/readyz` and a chat request with a demo key. `/metrics` must 404.
   Grafana: `ssh -L 3000:127.0.0.1:3000 ubuntu@<vm>` then http://localhost:3000.
8. **Monitoring.** Add an external uptime check (5 min) on `/healthz`.

### Continuous deploy

`.github/workflows/deploy.yml` runs after a successful `ci` run on `main` and on manual dispatch (optional `sha`
input for redeploys and rollbacks). It does nothing until these repository secrets exist:

| Secret / variable | Value |
|---|---|
| `DEPLOY_HOST` (secret) | VM public ip or dns name |
| `DEPLOY_SSH_KEY` (secret) | contents of the private key `gg-deploy` |
| `DEPLOY_KNOWN_HOSTS` (secret) | output of `ssh-keyscan -t ed25519 <vm>` |
| `DEPLOY_USER` (secret, optional) | defaults to `deploy` |
| `GG_SMOKE_KEY` (secret, optional) | the `smoke` key's token |
| `GG_PUBLIC_URL` (variable, optional) | `https://gg-demo.duckdns.org`, enables the outside checks |

The SSH key can only run `deploy.sh`, which accepts `deploy <sha>`, `restart` and `status`. `deploy` fetches,
checks out the sha, rebuilds the gateway image (`GIT_SHA` = sha), runs `docker compose up -d`, waits for `/version`
to report the sha and `/readyz` to pass, records it in `/opt/gg/.last_good`, and on failure redeploys the last good
sha and exits non-zero. A deploy costs a ~5-10 s gap while the single gateway container is recreated.

By hand: `ssh -i gg-deploy deploy@<vm> "deploy $(git rev-parse origin/main)"`.

## Fallback: Cloud Run

Use only if no A1 capacity is available or the VM is lost.

1. Build and push the amd64 image: `docker buildx build --platform linux/amd64 -t
   <region>-docker.pkg.dev/<project>/gg/gateway:<sha> --build-arg GIT_SHA=<sha> --push .`
2. Redis: a Redis Cloud free database (30 MB). Run `FT._LIST`; if the query engine is missing the semantic cache
   reports itself unavailable and exact caching keeps working.
3. Secrets: Secret Manager entries for the provider keys and `GG_REDIS_URL`; the keys file as a secret mounted at
   `/run/gg/keys.yaml`.
4. Models: a GCS bucket with the `gg-models` contents mounted read-write at `/opt/gg/models` (Cloud Storage volume
   mount), so cold starts do not re-download weights.
5. Deploy:

   ```sh
   gcloud run deploy gg --image <image> --port 8000 --cpu 1 --memory 2Gi --concurrency 40 \
     --min-instances 0 --max-instances 2 --timeout 300 --cpu-boost \
     --set-env-vars GG_MODEL_PROFILE=demo,GG_MODELS_DIR=/opt/gg/models,GG_KEYS_FILE=/run/gg/keys.yaml \
     --set-secrets GG_REDIS_URL=gg-redis-url:latest,GG_PROVIDERS__GEMINI__API_KEY=gg-gemini:latest
   ```

6. Metrics: Cloud Run has no Prometheus; scrape `/metrics` with a Grafana Alloy sidecar to Grafana Cloud free, or
   rely on Cloud Monitoring request metrics. Cold starts load the guard models, so the first request is slow;
   label live numbers with the host.
