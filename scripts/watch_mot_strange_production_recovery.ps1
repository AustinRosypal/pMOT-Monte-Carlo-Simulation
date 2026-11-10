param([Parameter(Mandatory=$true)][string]$Repository)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $Repository
$run = Join-Path $Repository 'outputs/statistics/mot_strange/symmetric_exact_D2_loading_20261105'
$cache = Join-Path $env:LOCALAPPDATA 'pmot-campaigns/symmetric_exact_D2_loading_20261105/sampled_production'
$launchPath = Join-Path $run 'production_launch.json'
$guard = [IO.File]::Open((Join-Path $cache 'recovery_guard.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
try {
    $old = Get-Content -LiteralPath $launchPath -Raw | ConvertFrom-Json -DateKind String
    $original = Get-CimInstance Win32_Process -Filter "ProcessId=$($old.uv_pid)"
    if ($original) {
        $expected = [DateTimeOffset]::Parse($old.uv_started_utc).UtcDateTime
        if ($original.CommandLine -ne $old.command -or
                [Math]::Abs(($original.CreationDate.ToUniversalTime() - $expected).TotalSeconds) -gt 1) {
            throw 'Production launch identity differs.'
        }
        Write-Output "Waiting for production runner $($original.ProcessId) to drain checkpoints."
        Wait-Process -Id $original.ProcessId
    }
    if (Get-CimInstance Win32_Process -Filter "ProcessId=$($old.python_pid) OR ParentProcessId=$($old.python_pid)") {
        throw 'Original production owner or workers are still present.'
    }
    $failurePath = Join-Path $cache 'failure.json'
    if (!(Test-Path -LiteralPath $failurePath)) {
        Write-Output 'Original production runner exited without the targeted error.'
        exit 0
    }
    $failure = Get-Content -LiteralPath $failurePath -Raw | ConvertFrom-Json
    $details = Get-Content -LiteralPath $old.stderr -Raw
    if ($failure.error -notmatch 'PermissionError' -or $details -notmatch 'progress\.json') {
        throw "Production exit needs independent review: $($failure.error)"
    }
    $tag = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
    Copy-Item -LiteralPath $launchPath -Destination (Join-Path $run "production_launch_before_$tag.json")
    $stdout = Join-Path $run "production_resumed_$tag.log"
    $stderr = Join-Path $run "production_resumed_$tag.err.log"
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:NUMBA_NUM_THREADS = '1'
    $launchOptions = @{
        FilePath = (Get-Command uv).Source
        ArgumentList = @('run','python','-u','scripts/resume_mot_strange_sampled_production.py','--workers',"$($old.workers)")
        WorkingDirectory = $Repository
        WindowStyle = 'Hidden'
        RedirectStandardOutput = $stdout
        RedirectStandardError = $stderr
        PassThru = $true
    }
    $runner = Start-Process @launchOptions
    Start-Sleep -Seconds 3
    $all = Get-CimInstance Win32_Process
    $uv = $all | Where-Object {$_.ProcessId -eq $runner.Id -and $_.CommandLine -like '*resume_mot_strange_sampled_production.py*'}
    $wrapper = $all | Where-Object {$_.ParentProcessId -eq $uv.ProcessId -and $_.CommandLine -like '*resume_mot_strange_sampled_production.py*'}
    $owner = $all | Where-Object {$_.ParentProcessId -eq $wrapper.ProcessId -and $_.CommandLine -like '*resume_mot_strange_sampled_production.py*'}
    if (!$uv -or !$wrapper -or !$owner) { throw "Production resume did not start. See $stderr" }
    $awakeArgs = '-NoProfile -ExecutionPolicy Bypass -File "{0}/scripts/keep_two_level_campaign_awake.ps1" -CampaignProcessId {1}' -f $Repository,$uv.ProcessId
    $helper = Start-Process -FilePath (Get-Command powershell.exe).Source -ArgumentList $awakeArgs -WindowStyle Hidden -PassThru
    $record = [ordered]@{
        uv_pid = $uv.ProcessId; wrapper_pid = $wrapper.ProcessId; python_pid = $owner.ProcessId
        uv_started_utc = $uv.CreationDate.ToUniversalTime().ToString('o')
        python_started_utc = $owner.CreationDate.ToUniversalTime().ToString('o')
        command = $uv.CommandLine; stdout = $stdout; stderr = $stderr
        awake_helper_pid = $helper.Id; workers = $old.workers
        original_failure = $failure.error
    }
    $temporary = Join-Path $run "production_launch_$tag.tmp"
    $record | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding utf8
    [IO.File]::Move($temporary, $launchPath, $true)
    $record | ConvertTo-Json
} finally {
    $guard.Dispose()
}
