[CmdletBinding()]
param(
    [string]$Branch = "pipeline/imagecond-mvp-20260715",
    [string]$RemoteName = "canondressgs-cloud",
    [string]$CloudHost = "canondress-cloud",
    [string]$BareRemote = "/root/autodl-tmp/canondressgs_work/remotes/canondressgs_pipeline_mvp.git",
    [string]$CloudWorktree = "/root/autodl-tmp/canondressgs_work/worktrees/canondressgs_pipeline_mvp",
    [string]$LogDirectory = "E:\model_train\CanonDressGS_Project\sync"
)

$ErrorActionPreference = "Stop"
$legacy = "/root/autodl-tmp/canondressgs_work/mmlphuman_code"
if ($CloudWorktree -eq $legacy -or -not $CloudWorktree.StartsWith("/root/autodl-tmp/canondressgs_work/worktrees/")) {
    throw "Refusing unsafe cloud target: $CloudWorktree"
}

$repoRoot = (git rev-parse --show-toplevel).Trim()
if ($LASTEXITCODE -ne 0) { throw "Not inside a Git repository." }
Set-Location $repoRoot
$currentBranch = (git branch --show-current).Trim()
$localCommit = (git rev-parse HEAD).Trim()
$localStatus = @(git status --porcelain)
if ($currentBranch -ne $Branch) { throw "Expected branch $Branch, found $currentBranch." }
if ($localStatus.Count -ne 0) { throw "Local worktree must be clean before deployment." }

$remoteUrl = "ssh://${CloudHost}${BareRemote}"
git remote get-url $RemoteName *> $null
if ($LASTEXITCODE -ne 0) { git remote add $RemoteName $remoteUrl }
elseif ((git remote get-url $RemoteName).Trim() -ne $remoteUrl) { throw "Remote $RemoteName points elsewhere." }

ssh $CloudHost "mkdir -p '$(Split-Path -Parent $BareRemote)' '$(Split-Path -Parent $CloudWorktree)'; test -d '$BareRemote' || git init --bare '$BareRemote'"
if ($LASTEXITCODE -ne 0) { throw "Failed to initialize cloud bare remote." }
git push $RemoteName "${Branch}:${Branch}"
if ($LASTEXITCODE -ne 0) { throw "Push failed." }

$cloudScript = "set -eu; if [ ! -d '$CloudWorktree/.git' ]; then git clone --branch '$Branch' '$BareRemote' '$CloudWorktree'; fi; cd '$CloudWorktree'; test -z `"`$(git status --porcelain)`"; git fetch origin '$Branch'; git checkout '$Branch'; git merge --ff-only 'origin/$Branch'; python3 -m py_compile train_dressable.py; printf '%s|%s|%s' `"`$(git rev-parse HEAD)`" `"`$(git branch --show-current)`" `"`$(git status --porcelain | wc -l)`""
$cloudState = (ssh $CloudHost $cloudScript).Trim()
if ($LASTEXITCODE -ne 0) { throw "Cloud checkout or import smoke failed." }
$remoteCommit = (git ls-remote $RemoteName "refs/heads/$Branch").Split("`t")[0]
$parts = $cloudState.Split("|")
if ($remoteCommit -ne $localCommit -or $parts[0] -ne $localCommit -or $parts[1] -ne $Branch -or $parts[2] -ne "0") {
    throw "Commit/state mismatch: local=$localCommit remote=$remoteCommit cloud=$cloudState"
}

New-Item -ItemType Directory -Force -Path $LogDirectory | Out-Null
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$record = [ordered]@{
    time = (Get-Date).ToString("o")
    branch = $Branch
    local_commit = $localCommit
    remote_commit = $remoteCommit
    cloud_checkout_commit = $parts[0]
    local_modified_count = $localStatus.Count
    cloud_modified_count = [int]$parts[2]
    cloud_worktree = $CloudWorktree
}
$record | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $LogDirectory "deploy_$stamp.json")
$record | ConvertTo-Json
