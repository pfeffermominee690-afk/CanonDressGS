[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][ValidatePattern('^[A-Za-z0-9._-]+$')][string]$RunId,
    [Parameter(Mandatory=$true)][string]$CloudRunDirectory,
    [string]$CloudHost = "canondress-cloud",
    [string]$DestinationRoot = "E:\model_train\CanonDressGS_Project\artifacts"
)

$ErrorActionPreference = "Stop"
if (-not $CloudRunDirectory.StartsWith("/root/autodl-tmp/canondressgs_work/outputs/")) {
    throw "Run directory must be under the external cloud outputs root."
}
$destination = Join-Path $DestinationRoot $RunId
New-Item -ItemType Directory -Force -Path $destination | Out-Null
$allowlist = @(
    "input_manifest.json", "manifest.json", "metrics.json", "metrics.md",
    "diagnostics.json", "diagnostics.md", "evaluation.log", "comparison.png"
)
$fetched = @()
foreach ($name in $allowlist) {
    ssh $CloudHost "test -f '$CloudRunDirectory/$name'"
    if ($LASTEXITCODE -eq 0) {
        scp "${CloudHost}:$CloudRunDirectory/$name" (Join-Path $destination $name)
        if ($LASTEXITCODE -ne 0) { throw "Failed to fetch $name" }
        $fetched += [ordered]@{ name=$name; sha256=(Get-FileHash -Algorithm SHA256 (Join-Path $destination $name)).Hash.ToLower() }
    }
}
$receipt = [ordered]@{
    time = (Get-Date).ToString("o")
    run_id = $RunId
    source = "${CloudHost}:$CloudRunDirectory"
    destination = $destination
    files = $fetched
}
$receipt | ConvertTo-Json -Depth 5 | Set-Content -Encoding utf8 (Join-Path $destination "FETCH_RECEIPT.json")
$receipt | ConvertTo-Json -Depth 5
