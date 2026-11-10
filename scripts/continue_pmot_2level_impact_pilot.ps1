param([Parameter(Mandatory=$true)][int]$UniformRunnerPid)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $repo
$output = Join-Path $repo 'outputs/statistics/pMOT_2Level/loading_20261109'
$uniform = Get-CimInstance Win32_Process -Filter "ProcessId=$UniformRunnerPid"
$progress = Get-Content -LiteralPath (Join-Path $output 'pilot_progress.json') -Raw | ConvertFrom-Json
if ($progress.pid -ne $UniformRunnerPid -or ($null -ne $uniform -and
        $uniform.CommandLine -notmatch 'run_pmot_2level_loading\.py pilot --workers 10')) {
    throw 'Uniform runner identity did not match this campaign.'
}
if ($null -ne $uniform) {
    Write-Output "Waiting for verified uniform pilot Python PID $UniformRunnerPid to finish."
    Wait-Process -Id $UniformRunnerPid -ErrorAction SilentlyContinue
}
$summary = Get-Content -LiteralPath (Join-Path $output 'pilot_summary.json') -Raw | ConvertFrom-Json
if ($summary.phase -ne 'pilot' -or $summary.seed -ne 2026110901 -or $summary.settings.Count -ne 240) {
    throw 'Uniform pilot did not produce its full summary.'
}
$impact = Join-Path $output 'impact_stratified'
$null = New-Item -Path $impact -ItemType Directory -Force
$launch = Join-Path $impact 'pilot_launch.json'
if (Test-Path -LiteralPath $launch) { throw 'Impact pilot has already been launched.' }
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$uv = (Get-Command uv -CommandType Application).Source
$job = Start-Process -FilePath $uv -ArgumentList @('run','python','-u','scripts/run_pmot_2level_impact_loading.py','pilot','--workers','10') `
    -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput (Join-Path $impact 'pilot.stdout.log') `
    -RedirectStandardError (Join-Path $impact 'pilot.stderr.log') -PassThru
[pscustomobject]@{
    uv_pid = $job.Id
    start_local = $job.StartTime.ToString('o')
    command = 'uv run python -u scripts/run_pmot_2level_impact_loading.py pilot --workers 10'
    uniform_python_pid = $UniformRunnerPid
} | ConvertTo-Json | Set-Content -LiteralPath $launch -Encoding utf8
Write-Output "Impact-stratified pilot started as uv PID $($job.Id)."
