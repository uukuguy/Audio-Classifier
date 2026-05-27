# CLAUDE.md — Fusion Control Competition

> Project-specific facts only. Global rules at `~/.claude/CLAUDE.md` apply.
> Lightweight-memory rules at `.claude/rules/lightweight-memory.md` (symlink to `~/.claude/shared-rules/lightweight-memory.md`) auto-load every session.

## Fresh clone setup

`.claude/` is gitignored, so the symlink doesn't survive clone:

```bash
mkdir -p .claude/rules
ln -s ~/.claude/shared-rules/lightweight-memory.md .claude/rules/lightweight-memory.md
```


## User private files (do not touch)

- `docs/MY-NOTE.md` — user's working scratch, do NOT delete/rename even if content looks duplicate. See `~/.claude/projects/.../memory/feedback_dont_touch_user_private_files.md`.
