<!-- ---
allowed-tools: Bash
description: >
  Extract file-level docstrings from all code files. Use this before loading
  individual files to understand the codebase structure and purpose. Supports
  optional path filter and --include-tests flag.
argument-hint: "[path-filter] [--include-tests] [--format text|json|markdown]"
---
 -->
# Get Project Docstrings

Run the docstring extraction tool and present a structured summary of what
every file does, so I can plan which files actually need to be read in full.

## Step 1 — Parse arguments

The user may pass:
- A path filter string (e.g. `src`, `mymodule/api`) — map to `-Filter`
- `--include-tests` flag — map to `-IncludeTests`
- `--format text|json|markdown` — map to `-OutputFormat`

Arguments received: `$ARGUMENTS`

## Step 2 — Run the script

Locate `get-docstrings.ps1`. Check these locations in order:
1. `.claude/scripts/get-docstrings.ps1` (project-local)
2. `scripts/get-docstrings.ps1`
3. `tools/get-docstrings.ps1`
4. The project root

Once found, build the PowerShell command. Use `pwsh` if available, else `powershell`.

Example construction logic:
```
$cmd = "pwsh -NoProfile -File <script_path>"
if filter arg provided: append "-Filter '<value>'"
if --include-tests:     append "-IncludeTests"
if --format provided:   append "-OutputFormat <value>"
else:                   append "-OutputFormat markdown"   # default for Claude
```

Run the command and capture output.

## Step 3 — Analyse and summarise

After the script output, provide:

1. **Quick orientation** (2–4 sentences): What is this codebase? What does it
   appear to do at a high level, based on the docstrings?

2. **Files missing docstrings**: List any files that returned "(no file-level
   docstring found)" — flag these as candidates to update.

3. **Recommended next steps**: Based on what you've read, suggest which files
   (if any) are worth reading in full to answer the user's likely question.
   Do NOT speculatively load files — wait for the user to confirm.

## Notes

- Default behaviour excludes `**/tests/**` and `**/test/**` paths.
- If the script is not found in any expected location, output a friendly error
  explaining where to place `get-docstrings.ps1` and show the expected directory
  structure:
  ```
  your-project/
  └── .claude/
      ├── commands/
      │   └── get-docstrings.md   ← this file
      └── scripts/
          └── get-docstrings.ps1  ← PowerShell script
  ```
- If `pwsh` and `powershell` are both unavailable, inform the user and suggest
  installing PowerShell 7+ from https://aka.ms/powershell
