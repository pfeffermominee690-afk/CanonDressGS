[CmdletBinding()]
param(
    [string]$Branch = "pipeline/imagecond-mvp-20260715",
    [string]$RemoteName = "canondressgs-cloud",
    [string]$CloudHost = "canondress-cloud",
    [string]$CloudWorktree = "/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_pipeline_mvp"
)

$ErrorActionPreference = "Stop"
$localBranch = (git branch --show-current).Trim()
$localHead = (git rev-parse HEAD).Trim()
$localDirty = @(git status --porcelain).Count
$remoteHead = "UNAVAILABLE"
git remote get-url $RemoteName *> $null
if ($LASTEXITCODE -eq 0) {
    $line = git ls-remote $RemoteName "refs/heads/$Branch"
    if ($LASTEXITCODE -eq 0 -and $line) { $remoteHead = $line.Split("`t")[0] }
}
$cloudRaw = (ssh $CloudHost "if [ -d '$CloudWorktree/.git' ]; then cd '$CloudWorktree' && printf '%s|%s|%s' `"`$(git rev-parse HEAD)`" `"`$(git branch --show-current)`" `"`$(git status --porcelain | wc -l)`"; else printf 'MISSING|MISSING|-1'; fi").Trim()
$cloud = $cloudRaw.Split("|")
$result = [ordered]@{
    expected_branch = $Branch
    local_branch = $localBranch
    local_head = $localHead
    local_modified_count = $localDirty
    remote_head = $remoteHead
    cloud_head = $cloud[0]
    cloud_branch = $cloud[1]
    cloud_modified_count = [int]$cloud[2]
    commits_match = ($localHead -eq $remoteHead -and $localHead -eq $cloud[0])
    states_clean = ($localDirty -eq 0 -and [int]$cloud[2] -eq 0)
}
$result | ConvertTo-Json
if (-not $result.commits_match -or -not $result.states_clean -or $localBranch -ne $Branch -or $cloud[1] -ne $Branch) { exit 2 }
