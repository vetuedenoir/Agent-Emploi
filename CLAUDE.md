<!-- rtk-instructions v2 -->
# RTK (Rust Token Killer) - Token-Optimized Commands

## Golden Rule

**Always prefix commands with `rtk`**. If RTK has a dedicated filter, it uses it. If not, it passes through unchanged. This means RTK is always safe to use.

**Important**: Even in command chains with `&&`, use `rtk`:
```bash
# ❌ Wrong
git add . && git commit -m "msg" && git push

# ✅ Correct
rtk git add . && rtk git commit -m "msg" && rtk git push
```

## Non-obvious behaviour

- `rtk --help` lists every available subcommand — consult it rather than guessing.
- Git passthrough works for ALL subcommands, even those `rtk --help` does not name.
- `rtk grep` runs raw (unfiltered) when given a format flag: `-c`, `-l`, `-L`, `-o`, `-Z`.
- Meta commands are called directly, never proxied: `rtk gain`, `rtk discover`, `rtk proxy <cmd>`.
<!-- /rtk-instructions -->
