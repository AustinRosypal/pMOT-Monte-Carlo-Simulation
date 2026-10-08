$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$campaignRoot = Join-Path $projectRoot 'outputs/statistics/mot_multilevel_population_rate_v1/cooling_diameter_20260930_fixed_intensity_baseline'
$launcherRecord = Join-Path $campaignRoot 'pilot_continuation_launcher.json'
@{ pid=$PID; status='waiting for retention audit'; started_at_utc=[DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content $launcherRecord
$savedAudit = Get-Content (Join-Path $campaignRoot 'retention_recheck/000/process.json') -Raw | ConvertFrom-Json
$auditLive = Get-CimInstance Win32_Process -Filter "ProcessId = $($savedAudit.pid)"
if ($auditLive) {
    if ($auditLive.CommandLine -notmatch 'audit_diameter_retention.py') { throw 'Audit PID was reused; refusing to wait or launch' }
    $auditHandle = Get-Process -Id $savedAudit.pid -ErrorAction SilentlyContinue
    if ($auditHandle) { $auditHandle.WaitForExit() }
}
$completed = Get-Content (Join-Path $campaignRoot 'retention_recheck/000/completion.json') -Raw | ConvertFrom-Json
if (-not $completed.complete) { throw 'Retention audit did not finish; inspect its logs' }
$activePools = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match '(run_population_diameter_campaign|audit_diameter_retention|collect_diameter_pilots).py'
}
if ($activePools) { throw 'Another campaign pool is active; refusing duplicate launch' }
if (Test-Path (Join-Path $campaignRoot 'pilot_collection.log')) { throw 'Existing collector log needs review before restarting' }
$collector = Start-Process -FilePath 'wsl.exe' -ArgumentList @('--exec','/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python','-u','scripts/collect_diameter_pilots.py','--start-index','1','--workers','16') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $campaignRoot 'pilot_collection.log') -RedirectStandardError (Join-Path $campaignRoot 'pilot_collection.stderr.log') -PassThru
@{ pid=$PID; status='collector launched'; collector_launcher_pid=$collector.Id; launched_at_utc=[DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content $launcherRecord
