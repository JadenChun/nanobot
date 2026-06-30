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
    "autoSync": false
  }
]
```

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
