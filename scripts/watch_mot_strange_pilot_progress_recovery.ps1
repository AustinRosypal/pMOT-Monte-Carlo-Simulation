param([Parameter(Mandatory=$true)][string]$Repository)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $Repository
$run = Join-Path $Repository 'outputs/statistics/mot_strange/symmetric_exact_D2_loading_20261105'
$launchPath = Join-Path $run 'pilot_launch.json'
$failurePath = Join-Path $run 'pilot_failure.json'
$guard = [IO.File]::Open((Join-Path $run 'pilot_recovery_guard.lock'),
    [IO.FileMode]::OpenOrCreate, [IO.FileAccess]::ReadWrite, [IO.FileShare]::None)
try {
    $old = Get-Content -LiteralPath $launchPath -Raw | ConvertFrom-Json -DateKind String
    if ($old.command -notlike '*scripts/run_mot_strange_loading_pilot.py*') {
        throw 'This guard applies only to the original pilot launcher.'
    }
    $original = Get-CimInstance Win32_Process -Filter "ProcessId=$($old.uv_pid)"
    if ($original) {
        $expected = [DateTimeOffset]::Parse($old.uv_started_utc).UtcDateTime
        if ($original.CommandLine -ne $old.command -or
                [Math]::Abs(($original.CreationDate.ToUniversalTime() - $expected).TotalSeconds) -gt 1) {
            throw 'Recorded original uv identity did not match.'
        }
        Write-Output "Waiting for original verified uv process $($original.ProcessId)."
        Wait-Process -Id $original.ProcessId
    }
    if (Get-CimInstance Win32_Process -Filter "ProcessId=$($old.python_pid) OR ParentProcessId=$($old.python_pid)") {
        throw 'The original Python owner or its workers are still present.'
    }
    if (!(Test-Path -LiteralPath $failurePath)) {
        Write-Output 'Original pilot exited without the targeted error. No restart was made.'
        exit 0
    }
    $failure = Get-Content -LiteralPath $failurePath -Raw | ConvertFrom-Json
    $details = Get-Content -LiteralPath $old.stderr -Raw
    if ($failure.error -notmatch 'PermissionError' -or $details -notmatch 'pilot_progress\.json') {
        throw "Original exit requires diagnosis: $($failure.error)"
    }
    $tag = [DateTime]::UtcNow.ToString('yyyyMMdd_HHmmss')
    Copy-Item -LiteralPath $launchPath -Destination (Join-Path $run "pilot_launch_before_$tag.json")
    Copy-Item -LiteralPath $failurePath -Destination (Join-Path $run "pilot_failure_before_$tag.json")
    $env:OPENBLAS_NUM_THREADS = '1'
    $env:OMP_NUM_THREADS = '1'
    $env:MKL_NUM_THREADS = '1'
    $env:NUMBA_NUM_THREADS = '1'
    $argsForLaunch = @{
        FilePath = (Get-Command uv).Source
        ArgumentList = @('run','python','-u','scripts/resume_mot_strange_loading_pilot.py','--workers',"$($old.workers)")
        WorkingDirectory = $Repository
        WindowStyle = 'Hidden'
        RedirectStandardOutput = Join-Path $run "pilot_resumed_$tag.log"
        RedirectStandardError = Join-Path $run "pilot_resumed_$tag.err.log"
        PassThru = $true
    }
    $runner = Start-Process @argsForLaunch
    Start-Sleep -Seconds 3
    $processes = Get-CimInstance Win32_Process
    $uv = $processes | Where-Object { $_.ProcessId -eq $runner.Id -and $_.CommandLine -like '*scripts/resume_mot_strange_loading_pilot.py*' }
    if (!$uv) { throw "Resume did not start. See $($argsForLaunch.RedirectStandardError)" }
    $wrapper = $processes | Where-Object { $_.ParentProcessId -eq $uv.ProcessId -and $_.CommandLine -like '*scripts/resume_mot_strange_loading_pilot.py*' }
    if (!$wrapper) { throw 'Resume Python wrapper is missing.' }
    $owner = $processes | Where-Object { $_.ParentProcessId -eq $wrapper.ProcessId -and $_.CommandLine -like '*scripts/resume_mot_strange_loading_pilot.py*' }
    if (!$owner) { throw 'Resume Python owner is missing.' }
    $awakeArgs = '-NoProfile -ExecutionPolicy Bypass -File "{0}/scripts/keep_two_level_campaign_awake.ps1" -CampaignProcessId {1}' -f $Repository,$uv.ProcessId
    $helper = Start-Process -FilePath (Get-Command powershell.exe).Source -ArgumentList $awakeArgs -WindowStyle Hidden -PassThru
    $record = [ordered]@{
        uv_pid = $uv.ProcessId; python_wrapper_pid = $wrapper.ProcessId; python_pid = $owner.ProcessId
        uv_started_utc = $uv.CreationDate.ToUniversalTime().ToString('o')
        python_started_utc = $owner.CreationDate.ToUniversalTime().ToString('o')
        command = $uv.CommandLine; stdout = $argsForLaunch.RedirectStandardOutput
        stderr = $argsForLaunch.RedirectStandardError; awake_helper_pid = $helper.Id; workers = $old.workers
        original_failure = $failure.error
        recovery = 'Progress publication retry and local mirror. Original manifest and checkpoints verified by the pilot runner.'
    }
    $temporary = Join-Path $run "pilot_launch_$tag.tmp"
    $record | ConvertTo-Json | Set-Content -LiteralPath $temporary -Encoding utf8
    [IO.File]::Move($temporary, $launchPath, $true)
    $record | ConvertTo-Json
} finally {
    $guard.Dispose()
}
