param([int]$Port = 8766, [switch]$LocalOnly, [switch]$NoBrowser)
$ErrorActionPreference = 'Stop'
$taskRoot = $PSScriptRoot
$taskRuntime = Join-Path $taskRoot '.runtime'
New-Item -ItemType Directory -Force -Path $taskRuntime | Out-Null
$taskServer = Join-Path $taskRoot 'server.py'
$taskPython = @('D:\python\python.exe', 'C:\Users\Administrator\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe') | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
if (-not $taskPython) { $taskPython = (Get-Command python -ErrorAction Stop).Source }
$taskUrl = "http://127.0.0.1:$Port/"
$taskExisting = $null
try { $taskExisting = Invoke-RestMethod -Uri ($taskUrl + 'api/health') -TimeoutSec 2 } catch {}
if ($taskExisting -and $taskExisting.app -ne '龙头战法本地看板') { throw 'Port is in use by a different application. Use start.ps1 -Port 8767.' }
if (-not $taskExisting) {
    $taskArgs = @('-u', ('"' + $taskServer + '"'), '--port', "$Port")
    if (-not $LocalOnly) { $taskArgs += '--lan' }
    $taskProcess = Start-Process -FilePath $taskPython -ArgumentList $taskArgs -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskRuntime 'server.log') -RedirectStandardError (Join-Path $taskRuntime 'server.err.log')
    @{pid=$taskProcess.Id;server=$taskServer;port=$Port;started_at=[DateTime]::UtcNow.ToString('o')} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskRuntime 'process.json') -Encoding UTF8
    for ($taskTry=0; $taskTry -lt 20; $taskTry++) {
        try { $taskExisting = Invoke-RestMethod -Uri ($taskUrl + 'api/health') -TimeoutSec 1; break } catch { Start-Sleep -Milliseconds 250 }
    }
    if (-not $taskExisting) { throw ('Service did not start. See ' + (Join-Path $taskRuntime 'server.err.log')) }
}
Write-Host ('Computer: ' + $taskUrl)
if (-not $LocalOnly) {
    $taskKey = Get-Content -LiteralPath (Join-Path $taskRuntime 'access_key.txt') -Raw
    foreach ($taskAddress in $taskExisting.lan) { Write-Host ('Phone (same LAN): http://' + $taskAddress + ':' + $Port + '/?key=' + $taskKey.Trim()) }
}
Write-Host 'Archived review + automatic free public data. Closing this console does not stop the server.'
Write-Host 'Use stop.ps1 to stop this dashboard only.'
if (-not $NoBrowser) { Start-Process $taskUrl }
