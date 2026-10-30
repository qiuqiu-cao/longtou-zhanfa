$ErrorActionPreference = 'Stop'
$taskRecordPath = Join-Path $PSScriptRoot '.runtime/process.json'
if (-not (Test-Path -LiteralPath $taskRecordPath)) { Write-Host 'No owned dashboard process recorded.'; exit }
$taskRecord = Get-Content -LiteralPath $taskRecordPath -Raw | ConvertFrom-Json
$taskProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $($taskRecord.pid)" -ErrorAction SilentlyContinue
if ($taskProcess -and $taskProcess.CommandLine -like ('*' + (Join-Path $PSScriptRoot 'server.py') + '*')) {
    Stop-Process -Id $taskRecord.pid
    Write-Host 'Dashboard stopped.'
} else { Write-Host 'Recorded process is absent or changed; nothing was stopped.' }
