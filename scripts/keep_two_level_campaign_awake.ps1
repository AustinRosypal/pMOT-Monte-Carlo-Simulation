param([Parameter(Mandatory=$true)][int]$CampaignProcessId)

# Prevent idle system sleep while the authorized long simulation is running.
# The display may still sleep. Windows releases this request when we exit.
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class MOTCampaignPowerRequest {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern uint SetThreadExecutionState(uint flags);
}
'@
$taskProcess = Get-Process -Id $CampaignProcessId -ErrorAction SilentlyContinue
if ($null -eq $taskProcess) { exit 0 }
$taskStart = $taskProcess.StartTime
try {
    [MOTCampaignPowerRequest]::SetThreadExecutionState([uint32]2147483649) | Out-Null
    while ($true) {
        $taskCurrent = Get-Process -Id $CampaignProcessId -ErrorAction SilentlyContinue
        if ($null -eq $taskCurrent -or $taskCurrent.StartTime -ne $taskStart) { break }
        Start-Sleep -Seconds 30
    }
} finally {
    [MOTCampaignPowerRequest]::SetThreadExecutionState([uint32]2147483648) | Out-Null
}
