<#
.SYNOPSIS
    Extracts file-level docstrings (top-of-file only) from all code files in a project.

.DESCRIPTION
    Scans code files and extracts the leading docstring/comment block at the top of
    each file (before any imports or code). Outputs filepath + docstring pairs so
    Claude can understand the codebase without loading every file in full.

    Supported formats:
      - Python       : """...""" or '''...''' at top of file (before imports)
      - JS/TS/C#/Java: /** ... */ or /* ... */ block comment at top
      - PowerShell   : <# ... #> block or # comment block at top
      - Bash/Shell   : # comment block at top
      - Ruby         : =begin...=end or # comment block
      - Go           : // comment block before package declaration

.PARAMETER Path
    Root path to scan. Defaults to current directory.

.PARAMETER Filter
    Glob-style path filter to restrict which files are scanned.
    Example: "src/**" or "mymodule/**"

.PARAMETER ExcludePatterns
    Array of path patterns to exclude. Defaults to tests, build artefacts, and
    vendor/dependency folders.

.PARAMETER IncludeTests
    Switch: if set, test folders (**/tests/, **/test/, **/__tests__/) are NOT excluded.

.PARAMETER Extensions
    Array of file extensions to scan. Defaults to common code file extensions.

.PARAMETER OutputFormat
    'text'  (default) — human-readable console output
    'json'  — machine-readable JSON array
    'markdown' — markdown with code fences (good for pasting into Claude)

.EXAMPLE
    # Scan entire project, skip tests (default)
    .\get-docstrings.ps1

.EXAMPLE
    # Scan only src/ folder
    .\get-docstrings.ps1 -Filter "src"

.EXAMPLE
    # Include tests, output as JSON
    .\get-docstrings.ps1 -IncludeTests -OutputFormat json

.EXAMPLE
    # Scan a specific path with custom exclusions
    .\get-docstrings.ps1 -Path "C:\myproject" -ExcludePatterns @("**/migrations/**","**/generated/**")
#>

param(
    [string]   $Path           = (Get-Location).Path,
    [string]   $Filter         = "",
    [string[]] $ExcludePatterns = @(),
    [switch]   $IncludeTests,
    [string[]] $Extensions     = @(
        ".py", ".js", ".ts", ".jsx", ".tsx",
        ".cs", ".java", ".go", ".rb", ".php",
        ".ps1", ".psm1", ".sh", ".bash",
        ".rs", ".kt", ".swift", ".cpp", ".c", ".h"
    ),
    [ValidateSet("text","json","markdown")]
    [string]   $OutputFormat   = "text"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# ── Default exclusion patterns ──────────────────────────────────────────────
$defaultExclusions = @(
    "**/node_modules/**",
    "**/.git/**",
    "**/dist/**",
    "**/build/**",
    "**/bin/**",
    "**/obj/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/venv/**",
    "**/env/**",
    "**/.env/**",
    "**/vendor/**",
    "**/packages/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/*.egg-info/**",
    "**/coverage/**",
    "**/.nyc_output/**"
)

if (-not $IncludeTests) {
    $defaultExclusions += @(
        "**/tests/**",
        "**/test/**",
        "**/__tests__/**",
        "**/spec/**",
        "**/*.test.*",
        "**/*.spec.*"
    )
}

$allExclusions = $defaultExclusions + $ExcludePatterns

# ── Helper: convert glob pattern to regex ────────────────────────────────────
function ConvertTo-GlobRegex([string]$glob) {
    $escaped = [regex]::Escape($glob)
    $escaped = $escaped -replace '\\\*\\\*/', '(.+/)?'   # **/ → any depth
    $escaped = $escaped -replace '\\\*',      '[^/]*'    # *  → any non-sep chars
    $escaped = $escaped -replace '\\\?',      '[^/]'     # ?  → single char
    return "(?i)$escaped"
}

# ── Helper: normalise path separators to forward slash ───────────────────────
function Normalize-Path([string]$p) {
    return $p.Replace('\','/')
}

# ── Helper: check if file path matches any exclusion pattern ────────────────
function IsExcluded([string]$fullPath) {
    $rel = Normalize-Path (Resolve-Path -Relative $fullPath -ErrorAction SilentlyContinue)
    if (-not $rel) { $rel = Normalize-Path $fullPath }
    foreach ($pattern in $allExclusions) {
        $rx = ConvertTo-GlobRegex $pattern
        if ($rel -match $rx) { return $true }
    }
    return $false
}

# ── Docstring extractors per language group ──────────────────────────────────

function Extract-PythonDocstring([string[]]$lines) {
    <#
    Captures the first triple-quoted string at the top of a Python file.
    Skips shebang lines and encoding comments but NOT regular code.
    #>
    $i = 0
    # Skip shebang / encoding lines
    while ($i -lt $lines.Count -and ($lines[$i] -match '^\s*#' -or $lines[$i] -match '^\s*$')) {
        $i++
    }
    if ($i -ge $lines.Count) { return $null }

    $line = $lines[$i]
    # Triple double-quote
    if ($line -match '^\s*("""|\x27\x27\x27)') {
        $delim  = $Matches[1]
        $result = @()
        $rest   = $line -replace "^\s*$([regex]::Escape($delim))",""

        # Single-line docstring: """text"""
        if ($rest -match "$([regex]::Escape($delim))\s*$") {
            return ($rest -replace "$([regex]::Escape($delim))\s*$","").Trim()
        }
        $result += $rest
        $i++
        while ($i -lt $lines.Count) {
            if ($lines[$i] -match $([regex]::Escape($delim))) {
                $result += ($lines[$i] -replace $([regex]::Escape($delim)),"")
                break
            }
            $result += $lines[$i]
            $i++
        }
        return ($result -join "`n").Trim()
    }
    return $null
}

function Extract-BlockComment([string[]]$lines, [string]$startPattern, [string]$endPattern) {
    <# Extracts /** ... */ or /* ... */ style block comments at top of file #>
    $i = 0
    # Skip blank lines and single-line // comments at the very top
    while ($i -lt $lines.Count -and $lines[$i] -match '^\s*$') { $i++ }
    if ($i -ge $lines.Count) { return $null }

    if ($lines[$i] -notmatch $startPattern) { return $null }

    $result = @()
    $firstLine = $lines[$i] -replace $startPattern,""
    if ($firstLine.Trim()) { $result += $firstLine.Trim() }
    $i++

    while ($i -lt $lines.Count) {
        if ($lines[$i] -match $endPattern) {
            $last = $lines[$i] -replace $endPattern,""
            if ($last.Trim()) { $result += $last.Trim() }
            break
        }
        # Strip leading * from /** */ style
        $cleaned = $lines[$i] -replace '^\s*\*\s?',""
        $result += $cleaned
        $i++
    }
    return ($result -join "`n").Trim()
}

function Extract-LineCommentBlock([string[]]$lines, [string]$commentChar) {
    <# Extracts a contiguous block of # or // comment lines at the top of file #>
    $i = 0
    while ($i -lt $lines.Count -and $lines[$i] -match '^\s*$') { $i++ }
    # Skip shebangs
    if ($i -lt $lines.Count -and $lines[$i] -match '^#!') { $i++ }
    while ($i -lt $lines.Count -and $lines[$i] -match '^\s*$') { $i++ }

    $result = @()
    $rx = "^\s*$([regex]::Escape($commentChar))\s?"
    while ($i -lt $lines.Count -and $lines[$i] -match "^\s*$([regex]::Escape($commentChar))") {
        $result += ($lines[$i] -replace $rx,"")
        $i++
    }
    if ($result.Count -eq 0) { return $null }
    return ($result -join "`n").Trim()
}

function Extract-PowerShellBlock([string[]]$lines) {
    <# Extracts <# ... #> block at top, or falls back to # comment block #>
    $i = 0
    while ($i -lt $lines.Count -and $lines[$i] -match '^\s*$') { $i++ }
    if ($i -ge $lines.Count) { return $null }

    if ($lines[$i] -match '^\s*<#') {
        $result = @()
        $rest = $lines[$i] -replace '^\s*<#',""
        if ($rest.Trim()) { $result += $rest.Trim() }
        $i++
        while ($i -lt $lines.Count) {
            if ($lines[$i] -match '#>') {
                $last = $lines[$i] -replace '#>',""
                if ($last.Trim()) { $result += $last.Trim() }
                break
            }
            $result += $lines[$i].TrimStart()
            $i++
        }
        return ($result -join "`n").Trim()
    }
    # Fall back to # comment block
    return Extract-LineCommentBlock $lines "#"
}

function Extract-GoDocstring([string[]]$lines) {
    <# Go: // comment block immediately before 'package' declaration #>
    $result = @()
    foreach ($line in $lines) {
        if ($line -match '^\s*package\s+\w+') { break }
        if ($line -match '^\s*//') {
            $result += ($line -replace '^\s*//\s?',"")
        } elseif ($line -match '^\s*$' -and $result.Count -gt 0) {
            continue
        } elseif ($result.Count -gt 0) {
            break
        }
    }
    if ($result.Count -eq 0) { return $null }
    return ($result -join "`n").Trim()
}

function Get-Docstring([string]$filePath) {
    try {
        $lines = [System.IO.File]::ReadAllLines($filePath)
    } catch {
        return $null
    }

    $ext = [System.IO.Path]::GetExtension($filePath).ToLower()

    switch ($ext) {
        { $_ -in @(".py") } {
            return Extract-PythonDocstring $lines
        }
        { $_ -in @(".js",".ts",".jsx",".tsx",".java",".cs",".php",".kt",".swift") } {
            $result = Extract-BlockComment $lines '^\s*/\*[*!]?' '\*/'
            if (-not $result) {
                $result = Extract-LineCommentBlock $lines "//"
            }
            return $result
        }
        { $_ -in @(".go") } {
            return Extract-GoDocstring $lines
        }
        { $_ -in @(".ps1",".psm1") } {
            return Extract-PowerShellBlock $lines
        }
        { $_ -in @(".sh",".bash",".rb") } {
            return Extract-LineCommentBlock $lines "#"
        }
        { $_ -in @(".cpp",".c",".h",".rs") } {
            $result = Extract-BlockComment $lines '^\s*/\*[*!]?' '\*/'
            if (-not $result) {
                $result = Extract-LineCommentBlock $lines "//"
            }
            return $result
        }
        default {
            return Extract-LineCommentBlock $lines "#"
        }
    }
}

# ── Collect files ─────────────────────────────────────────────────────────────
Push-Location $Path

$allFiles = Get-ChildItem -Path $Path -Recurse -File |
    Where-Object { $Extensions -contains $_.Extension.ToLower() }

if ($Filter) {
    $filterRx = ConvertTo-GlobRegex ("**/$Filter/**")
    # Also match if filter appears anywhere in the relative path
    $allFiles = $allFiles | Where-Object {
        $rel = Normalize-Path (Resolve-Path -Relative $_.FullName -ErrorAction SilentlyContinue)
        $rel -match $filterRx -or $rel -match [regex]::Escape($Filter)
    }
}

$allFiles = $allFiles | Where-Object { -not (IsExcluded $_.FullName) }

# ── Extract docstrings ────────────────────────────────────────────────────────
$results = @()
foreach ($file in $allFiles) {
    $rel      = Normalize-Path (Resolve-Path -Relative $file.FullName -ErrorAction SilentlyContinue)
    if (-not $rel) { $rel = Normalize-Path $file.FullName }
    $docstring = Get-Docstring $file.FullName

    $results += [PSCustomObject]@{
        FilePath  = $rel -replace '^\./','.'
        Docstring = if ($docstring) { $docstring } else { "(no file docstring found)" }
        HasDoc    = [bool]$docstring
    }
}

Pop-Location

# ── Output ────────────────────────────────────────────────────────────────────
switch ($OutputFormat) {
    "json" {
        $results | ConvertTo-Json -Depth 3
    }
    "markdown" {
        Write-Output "# Project Docstring Index`n"
        Write-Output "> Generated by get-docstrings.ps1 | $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
        Write-Output "> Scanned: $Path"
        if (-not $IncludeTests) { Write-Output "> Tests excluded (use -IncludeTests to include)" }
        Write-Output ""
        $withDoc    = ($results | Where-Object { $_.HasDoc }).Count
        $withoutDoc = ($results | Where-Object { -not $_.HasDoc }).Count
        Write-Output "**Files with docstrings:** $withDoc  |  **Missing docstrings:** $withoutDoc  |  **Total:** $($results.Count)`n"
        Write-Output "---`n"
        foreach ($r in $results | Sort-Object FilePath) {
            Write-Output "### ``$($r.FilePath)```n"
            if ($r.HasDoc) {
                Write-Output "``````"
                Write-Output $r.Docstring
                Write-Output "``````"
            } else {
                Write-Output "_No file-level docstring found._"
            }
            Write-Output ""
        }
    }
    default {
        # text
        $sep = "=" * 72
        Write-Output $sep
        Write-Output "  PROJECT DOCSTRING INDEX"
        Write-Output "  Path    : $Path"
        Write-Output "  Scanned : $(Get-Date -Format 'yyyy-MM-dd HH:mm')"
        if (-not $IncludeTests) { Write-Output "  Tests   : excluded  (use -IncludeTests to include)" }
        $withDoc = ($results | Where-Object { $_.HasDoc }).Count
        Write-Output "  Files   : $($results.Count) total  |  $withDoc with docstrings  |  $($results.Count - $withDoc) missing"
        Write-Output $sep
        Write-Output ""
        foreach ($r in $results | Sort-Object FilePath) {
            Write-Output "FILE: $($r.FilePath)"
            Write-Output ("-" * ([Math]::Min(72, $r.FilePath.Length + 6)))
            if ($r.HasDoc) {
                Write-Output $r.Docstring
            } else {
                Write-Output "(no file-level docstring found)"
            }
            Write-Output ""
        }
        Write-Output $sep
        Write-Output "  END OF INDEX  |  $($results.Count) files scanned"
        Write-Output $sep
    }
}
