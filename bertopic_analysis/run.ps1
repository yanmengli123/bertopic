param([switch]$Offline, [switch]$ReportOnly)
$ErrorActionPreference = 'Stop'
$analysisPython = Join-Path (Split-Path -Parent $PSScriptRoot) '.venv_bertopic\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $analysisPython)) {
    throw 'Python environment missing. Follow README.md to create it.'
}
$env:USE_TORCH = '0'
$env:USE_TF = '0'
$analysisArguments = @((Join-Path $PSScriptRoot 'run_pipeline.py'))
if ($Offline) { $analysisArguments += '--offline' }
if ($ReportOnly) { $analysisArguments += '--report-only' }
& $analysisPython @analysisArguments
if ($LASTEXITCODE -ne 0) { throw "Pipeline failed: $LASTEXITCODE" }
