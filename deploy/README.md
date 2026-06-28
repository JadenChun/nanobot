# Nanobot Gateway Deployment

This deploys the marketing-agent Nanobot gateway on the Lightsail instance.

Expected paths:

```text
/opt/marketing-agent/nanobot-custom
/opt/marketing-agent/client-marketing-assistance
/opt/marketing-agent/secrets/nanobot-config.json
/opt/marketing-agent/secrets/nanobot.env
```

The secret files should be owned by `root:marketing-agent` with mode `0640` so
systemd and the `marketing-agent` service user can read them.

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
sudoedit /opt/marketing-agent/secrets/nanobot-config.json
sudoedit /opt/marketing-agent/secrets/nanobot.env
sudo systemctl enable --now nanobot-gateway
```

Required values:

- Telegram bot token in `nanobot-config.json`.
- Telegram `allowFrom` or `allowChats` in `nanobot-config.json`.
- Provider API key in `nanobot-config.json`, or through `NANOBOT_*` env vars.
- `OAUTH_BROKER_INTERNAL_API_KEY` in `nanobot.env`; it must match the OAuth
  broker's internal API key.

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
