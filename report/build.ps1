# Build the thesis with MiKTeX without requiring Perl or a refreshed VS Code PATH.
[CmdletBinding()]
param(
    [string]$MainFile
)

Set-StrictMode -Version 2.0
$ErrorActionPreference = 'Stop'

# Resolve the default after parameter binding for Windows PowerShell 5.1.
if ([string]::IsNullOrWhiteSpace($MainFile)) {
    $MainFile = Join-Path $PSScriptRoot 'main.tex'
}

# LaTeX Workshop's %DOC% placeholder omits the .tex extension.
if ([System.IO.Path]::GetExtension($MainFile) -eq '') {
    $MainFile += '.tex'
}
if (-not (Test-Path -LiteralPath $MainFile -PathType Leaf)) {
    throw "Main LaTeX file not found: $MainFile"
}
$resolvedMainFile = (Resolve-Path -LiteralPath $MainFile).ProviderPath

# Prefer an existing installation, then try the standard MiKTeX directories.
$compilerPath = $null
$compilerCommand = Get-Command 'texify.exe' -CommandType Application -ErrorAction SilentlyContinue |
    Select-Object -First 1
if ($null -ne $compilerCommand) {
    $compilerPath = $compilerCommand.Source
}
if ([string]::IsNullOrWhiteSpace($compilerPath)) {
    $candidatePaths = @()
    if (-not [string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        $candidatePaths += Join-Path $env:LOCALAPPDATA 'Programs\MiKTeX\miktex\bin\x64\texify.exe'
    }
    if (-not [string]::IsNullOrWhiteSpace($env:ProgramFiles)) {
        $candidatePaths += Join-Path $env:ProgramFiles 'MiKTeX\miktex\bin\x64\texify.exe'
    }
    foreach ($candidatePath in $candidatePaths) {
        if (Test-Path -LiteralPath $candidatePath -PathType Leaf) {
            $compilerPath = $candidatePath
            break
        }
    }
}
if ([string]::IsNullOrWhiteSpace($compilerPath)) {
    throw 'MiKTeX texify.exe was not found. Install MiKTeX or add its executable directory to PATH.'
}

$compilerArguments = @(
    '--pdf'
    '--batch'
    '--tex-option=-synctex=1'
    '--tex-option=-interaction=nonstopmode'
    '--tex-option=-file-line-error'
    '--tex-option=-halt-on-error'
    [System.IO.Path]::GetFileName($resolvedMainFile)
)

# Child tools such as pdflatex and bibtex must see the same MiKTeX directory.
$previousPath = $env:Path
$compilerDirectory = Split-Path -Path $compilerPath -Parent
$env:Path = $compilerDirectory + [System.IO.Path]::PathSeparator + $previousPath
$locationPushed = $false
$exitCode = 1
try {
    Push-Location -LiteralPath (Split-Path -Path $resolvedMainFile -Parent)
    $locationPushed = $true
    Write-Host "Building $resolvedMainFile with $compilerPath"
    & $compilerPath @compilerArguments
    $exitCode = $LASTEXITCODE
}
catch {
    Write-Error -Message $_.Exception.Message -ErrorAction Continue
    $exitCode = 1
}
finally {
    if ($locationPushed) {
        Pop-Location
    }
    $env:Path = $previousPath
}

exit $exitCode
