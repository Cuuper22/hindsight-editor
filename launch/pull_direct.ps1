param([string[]]$Waves)
# Pull run metrics straight from live Colab VMs (sidesteps the HF repo when its commit API rate-limits).
Set-Location (Split-Path $PSScriptRoot -Parent)
foreach ($w in $Waves) {
  $c = "runs\_colab\$w.json"
  if (-not (Test-Path $c)) { "no session file for $w"; continue }
  $s = (Get-Content $c | ConvertFrom-Json).PSObject.Properties.Name | Select-Object -First 1
  $dirs = colab --config $c ls /content/runs -s $s 2>$null | Where-Object { $_ -match '^r_' }
  foreach ($d in $dirs) {
    $rid = $d.TrimEnd('/')
    New-Item -ItemType Directory -Force "runs\$rid" | Out-Null
    $files = colab --config $c ls "/content/runs/$rid" -s $s 2>$null
    foreach ($f in @('metrics.jsonl', 'summary.json', 'DONE', 'config.json')) {
      if ($files -contains $f) { colab --config $c download "/content/runs/$rid/$f" "runs\$rid\$f" -s $s 2>$null | Out-Null }
    }
    foreach ($e in ($files | Where-Object { $_ -match '^eval_s' })) {
      $ed = $e.TrimEnd('/'); New-Item -ItemType Directory -Force "runs\$rid\$ed" | Out-Null
      foreach ($f in @('metrics.jsonl', 'summary.json')) { colab --config $c download "/content/runs/$rid/$ed/$f" "runs\$rid\$ed\$f" -s $s 2>$null | Out-Null }
    }
    "$w  $rid"
  }
}
