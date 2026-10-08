$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$campaignRoot = Join-Path $projectRoot 'outputs/statistics/mot_multilevel_population_rate_v1/cooling_diameter_20260930_fixed_intensity_baseline'
$launcherRecord = Join-Path $campaignRoot 'fixed_validation_launcher.json'
@{ pid=$PID; status='waiting for pilot collector'; started_at_utc=[DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content $launcherRecord
$saved = Get-Content (Join-Path $campaignRoot 'pilot_collection_process.json') -Raw | ConvertFrom-Json
$live = Get-CimInstance Win32_Process -Filter "ProcessId = $($saved.pid)"
if ($live) {
    if ($live.CommandLine -notmatch 'collect_diameter_pilots.py') { throw 'Collector PID reused; refusing launch' }
    $handle = Get-Process -Id $saved.pid -ErrorAction SilentlyContinue
    if ($handle) { $handle.WaitForExit() }
}
if (-not (Test-Path (Join-Path $campaignRoot 'pilot_collection_complete.json'))) { throw 'Pilot collection did not complete' }
$activePools = Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match '^python' -and $_.CommandLine -match '(run_population_diameter_campaign|audit_diameter_retention|collect_diameter_pilots|validate_diameter_pilot_masks).py'
}
if ($activePools) { throw 'Campaign pool active; refusing duplicate launch' }
if (Test-Path (Join-Path $campaignRoot 'fixed_validation.log')) { throw 'Existing validation log needs review' }
$child = Start-Process -FilePath 'wsl.exe' -ArgumentList @('--exec','/home/ajrosy/pMOT_MonteCarlo/.venv_pMOT_MC/bin/python','-u','scripts/validate_diameter_pilot_masks.py') -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $campaignRoot 'fixed_validation.log') -RedirectStandardError (Join-Path $campaignRoot 'fixed_validation.stderr.log') -PassThru
@{ pid=$PID; status='validator launched'; child_launcher_pid=$child.Id; launched_at_utc=[DateTime]::UtcNow.ToString('o') } | ConvertTo-Json | Set-Content $launcherRecord
