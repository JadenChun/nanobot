# desktop-use

Drive the local desktop directly from Python — no AX tree, no daemon, no
session. Every call is one OS-level event (CGEvent on macOS, planned for
Windows/Linux). Pair with `screenshot` to see what you're doing.

## When to use

Use `desktop_use` when you need pixel-precise control over arbitrary GUI
apps and you want a tool that *just works* without bootstrap permissions
beyond Accessibility + Screen Recording.

## Enable

In `~/.nanobot/config.json`:

```json
{
  "tools": {
    "desktop_use": { "enabled": true }
  }
}
```

Install the macOS backend extras once:

```bash
pip install 'nanobot-ai[desktop]'
```

Then in **System Settings -> Privacy & Security**, grant your terminal (or
the Python binary) both **Accessibility** and **Screen Recording**.

## Action vocabulary

| Action | Required args | Notes |
|---|---|---|
| `info` | — | Returns native screen size + cursor pos. Call first. |
| `screenshot` | — | Captures screen, returns inline image + scaled size. |
| `cursor_position` | — | Returns current cursor in scaled coords. |
| `mouse_move` | `coordinate=[x,y]` | Move only, no click. |
| `left_click` / `right_click` / `middle_click` / `double_click` | `coordinate=[x,y]` (optional) | If coord omitted, clicks at current cursor. |
| `left_click_drag` | `start_coordinate=[x,y]`, `coordinate=[x,y]` | Press, drag, release. |
| `scroll` | `scroll_direction`, `scroll_amount` (1-100), `coordinate` (optional) | Wheel notches. |
| `type` | `text` | Unicode injection — works for any character. |
| `key` | `text="Cmd+Shift+S"` | Modifiers: `Cmd`, `Ctrl`, `Alt`/`Opt`, `Shift`, `Fn`. |
| `hold_key` | `text`, `duration` (sec, <=10) | |
| `wait` | `duration` (sec, <=30) | |

## Coordinate space

Coordinates are in the **scaled screenshot space** the tool returns, not
native screen pixels. The tool downscales to one of XGA / WXGA / FWXGA so
the model sees screenshots at a familiar size. The wrapper converts back
to native pixels before dispatching OS events.

If you ever need native coords (e.g. for window-management debugging), set
`scaling_enabled: false` in the config block.

## Typical loop

```
1. info                               -> learn screen size
2. screenshot                         -> see desktop
3. mouse_move coordinate=[420, 300]   -> hover button (visible cursor travel)
4. left_click                         -> click at current pos
5. screenshot                         -> verify result
6. type text="hello"                  -> type into focused field
7. key text="Cmd+S"                   -> save
```

## Failure modes

- *"CGWindowListCreateImage returned None"* — Screen Recording permission
  not granted to the binary that's running nanobot.
- *"unknown key: …"* — Unrecognized key name in a chord; use a single
  character or a name from the table in `desktop_use.py::_VK`.
- Clicks that "do nothing" — Accessibility permission not granted; events
  are posted but the OS drops them silently.
