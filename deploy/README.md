# Nanobot Gateway Deployment

This deploys the marketing-agent Nanobot gateway on the Lightsail instance.

Expected paths:

```text
/opt/marketing-agent/nanobot-custom
/opt/marketing-agent/client-marketing-assistance
/opt/marketing-agent/.nanobot/config.json
/opt/marketing-agent/secrets/nanobot.env
```

`nanobot.env` should be owned by `root:marketing-agent` with mode `0640` so
systemd and the `marketing-agent` service user can read it. The Nanobot config
lives under `.nanobot` because Nanobot derives runtime directories from the
config file parent and needs to write cron, media, and credential state there.

`nanobot-config.json` attaches the context repo through:

```json
"contextRepos": [
  {
    "path": "/opt/marketing-agent/client-marketing-assistance",
    "readOnly": false,
    "autoSync": true
  }
]
```

Keep `autoSync` enabled so generated Markdown memory and other managed context
updates are committed and pushed to the context repository's configured Git
upstream. The repository manifest's `sync.autoPush` setting cannot enable sync
when this runtime flag is false. The deploy script now migrates the existing
marketing context entry to `autoSync: true`; it does not overwrite tokens or
other runtime configuration.

`nanobot.env` provides runtime environment for the context repo scripts,
including `OAUTH_BROKER_INTERNAL_URL` and `OAUTH_BROKER_INTERNAL_API_KEY`.

## First Install

Clone the Nanobot repo into `/opt/marketing-agent/nanobot-custom`, then run:

```bash
sudo bash /opt/marketing-agent/nanobot-custom/deploy/scripts/update_nanobot_gateway.sh
sudoedit /opt/marketing-agent/.nanobot/config.json
sudoedit /opt/marketing-agent/secrets/nanobot.env
sudo systemctl enable --now nanobot-gateway
```

Required values:

- Telegram bot token in `/opt/marketing-agent/.nanobot/config.json`.
- Telegram `allowFrom` or `allowChats` in `/opt/marketing-agent/.nanobot/config.json`.
- Provider API key in `/opt/marketing-agent/.nanobot/config.json`, or through `NANOBOT_*` env vars.
- `OAUTH_BROKER_INTERNAL_API_KEY` in `nanobot.env`; it must match the OAuth
  broker's internal API key.

## Multiple Cron Destinations

Cron jobs keep `payload.channel` and `payload.to` as their primary execution
and delivery context. To deliver the same completed result to other chats
without running the workflow again, add `additionalDestinations` to the job in
`/opt/marketing-agent/.nanobot/workspace/cron/jobs.json`:

```json
{
  "payload": {
    "deliver": true,
    "channel": "telegram",
    "to": "PRIMARY_DM_CHAT_ID",
    "additionalDestinations": [
      {
        "channel": "telegram",
        "to": "GROUP_CHAT_ID"
      }
    ]
  }
}
```

Each destination is de-duplicated before delivery. Telegram groups must still
be approved through `channels.telegram.allowChats` in `config.json`.

## Scheduled Marketing Messages

Keep `channels.taskUpdateMode` set to `result`. Scheduled jobs execute their
full workflow internally and send one completed response to each configured
destination. They do not send progress updates or report attachments unless a
job explicitly requires an attachment. Job prompts must request clean Unicode
text and must not copy broken encoding artifacts from source material.

Update the production cron store so the recurring marketing jobs use these
contracts:

- **Daily trend research** (`0 6 * * *`, Asia/Kuala_Lumpur): research a rolling
  window of approximately the previous 14 days, not only signals published in
  the last day. Cover cat dried food, freeze-dried food, and tofu litter across
  TikTok, Instagram, Threads, Facebook, YouTube, Google Trends, Reddit, and
  relevant EGOCAT/pet-owner discussions. Compare new signals with recent memory
  to distinguish emerging momentum, repeated themes, weakening signals, and
  one-off noise. Cite and date the source of every included signal. Return a
  short Telegram summary grouped as `MAKE`, `CONSIDER`, and `SKIP`.
- **Daily content idea** (`0 8 * * *`, Asia/Kuala_Lumpur): replace the Monday
  weekly content batch. Before generating an idea, synthesize trend-research
  memory and performance/review evidence from approximately the previous 14
  days. Do not rely only on the newest research file or most recent review.
  Generate one immediately usable idea using judgment across repeated patterns,
  momentum, recency, source quality, and performance learning. Label its pillar
  as Education, Humour, Product, Awareness, Community, or Trends; keep it
  limited to cat dried food, freeze-dried food, or tofu litter; include the
  angle, hook, execution, CTA, the trend evidence used with source link(s), and
  the performance learning applied directly in the Telegram response. If the
  evidence window is incomplete or stale, state that limitation instead of
  inventing evidence.
- **Weekly performance summary** (retain the existing weekly schedule): analyze
  available performance evidence across approximately the previous 14 days,
  rather than treating only the newest day or run as representative. Compare
  posts and repeated patterns across that window, while weighting recent and
  sufficiently sampled results appropriately. Return `Best performer`, `Worst
  performer`, `Key learning`, and `Recommended action`, followed by `What
  happened?`, `Why?`, and `What should we do next?`. Keep the response concise
  and do not attach or direct the user to a separate report.

Continue writing internal research, analysis, idea, and performance-review
memory when it is useful to future runs. Use compact Markdown rather than the
previous long user-facing report format. Keep only:

- durable evidence and useful source links;
- important findings or patterns;
- decisions and their reasons;
- the operational outcome;
- the next improvement, action, or experiment.

Do not retain raw research trails, repeated explanations, or every intermediate
observation. These files are improvement-loop memory for the agent, not client
deliverables. Retain enough dated evidence for at least the rolling 14-day
analysis window. Internal persistence remains separate from what Telegram
sends.

For low-latency marketing operations, the example config maps common Telegram
slash commands such as `/connection_status` and `/sync_meta_analytics` directly
to context-repo scripts. Commands not listed in `fastCommands` still go through
the normal agent loop.

## Updates

Use:

```bash
sudo bash /opt/marketing-agent/nanobot-custom/deploy/scripts/update_nanobot_gateway.sh
```

The update script pulls the configured branch and only redeploys/restarts the
gateway when Nanobot code changed, or when the service is not active.

## Operations

```bash
sudo systemctl status nanobot-gateway --no-pager
sudo journalctl -u nanobot-gateway -n 100 --no-pager
sudo tail -n 100 /opt/marketing-agent/logs/nanobot-gateway.log
```

To verify context-memory delivery after a scheduled run:

```bash
sudo -u marketing-agent git -C /opt/marketing-agent/client-marketing-assistance status --short --branch
sudo -u marketing-agent git -C /opt/marketing-agent/client-marketing-assistance rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
sudo grep -E 'Context repo (synced successfully|sync did not complete|sync failed)' /opt/marketing-agent/logs/nanobot-gateway.log | tail -n 20
```

A successful scheduled run waits for its context sync attempt before completing.
Interactive requests keep the sync in the background to avoid delaying the chat
response. Confirm that the service user has non-interactive push access to the
configured upstream; `autoSync` cannot compensate for missing GitHub credentials.

## Optional Crawl4AI Worker

The Crawl4AI integration keeps browser execution outside the main agent process:

```text
Nanobot orchestrator -> lower-cost foreground crawler role -> social_crawl -> Unix socket -> Crawl4AI/Chromium
```

The foreground crawler role receives rendered, cleaned HTML with links and DOM attributes intact.
JSON is used only on the private Unix socket. The `social_crawl` tool unwraps it and gives the
role a short session/URL preamble followed by the HTML directly. Crawl4AI does not receive
an LLM key and does not perform reasoning itself.

Install the isolated worker only after the normal gateway is healthy:

```bash
sudo bash /opt/marketing-agent/nanobot-custom/deploy/scripts/deploy_crawl4ai_worker.sh
sudo journalctl -u crawl4ai-worker -n 100 --no-pager
```

Check it locally through its Unix socket:

```bash
sudo -u marketing-agent python3 - <<'PY'
import json
import socket

client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
client.connect("/run/crawl4ai-worker/worker.sock")
client.sendall(b'{"op":"health"}\n')
print(json.loads(client.makefile().readline()))
PY
```

Then set both controls before restarting Nanobot:

1. Set `agents.crawler.enabled` to `true` in
   `/opt/marketing-agent/.nanobot/config.json`, then choose its provider and lower-cost model.
   If this differs from the main agent provider, configure that provider's API key too.
2. Set `CRAWL4AI_WORKER_ENABLED=true` in
   `/opt/marketing-agent/secrets/nanobot.env`.

Finally restart the gateway:

```bash
sudo systemctl restart nanobot-gateway
```

The first rollout should keep the worker headless, logged out, and limited to the domains in
`/etc/crawl4ai-worker/crawl4ai-worker.env`. It serializes browser actions and allows one live
session, which is intentional for the 2 GB Lightsail instance. TikTok remains outside the
initial scope. The worker accepts only open, inspect, CSS-selector click, scroll, wait, and
close operations; it never accepts arbitrary JavaScript, cookies, credentials, or proxy
settings from the agent.
