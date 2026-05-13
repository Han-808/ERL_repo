param(
  [Parameter(Mandatory = $true)]
  [string]$Csv,

  [Parameter(Mandatory = $true)]
  [string]$Out,

  [string]$Title = "Sample Accuracy Distribution by State",

  [string]$Env = "",

  [int]$Width = 1320,

  [int]$PanelHeight = 360
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function SvgEscape([string]$Text) {
  return [System.Security.SecurityElement]::Escape($Text)
}

function EnvLabel([string]$EnvName) {
  switch ($EnvName) {
    "frozen_lake" { return "FrozenLake" }
    "sokoban" { return "Sokoban" }
    default { return $EnvName }
  }
}

function Mean($Values) {
  $arr = @($Values | ForEach-Object { [double]$_ })
  if ($arr.Count -eq 0) { return 0.0 }
  return ($arr | Measure-Object -Average).Average
}

function Quantile($Values, [double]$Q) {
  $arr = @($Values | ForEach-Object { [double]$_ } | Sort-Object)
  if ($arr.Count -eq 0) { return 0.0 }
  if ($arr.Count -eq 1) { return $arr[0] }
  $pos = ($arr.Count - 1) * $Q
  $lo = [math]::Floor($pos)
  $hi = [math]::Ceiling($pos)
  if ($lo -eq $hi) { return $arr[$lo] }
  $w = $pos - $lo
  return $arr[$lo] * (1.0 - $w) + $arr[$hi] * $w
}

function KernelDensity($Values, [double]$Y, [double]$Bandwidth) {
  $sum = 0.0
  foreach ($v in $Values) {
    $z = ($Y - [double]$v) / $Bandwidth
    $sum += [math]::Exp(-0.5 * $z * $z)
  }
  return $sum
}

function ToFixed([double]$Value, [int]$Digits = 3) {
  return $Value.ToString("F$Digits", [Globalization.CultureInfo]::InvariantCulture)
}

if (-not (Test-Path -LiteralPath $Csv)) {
  throw "CSV not found: $Csv"
}

$rows = @(Import-Csv -LiteralPath $Csv)
$required = @("env", "state_x", "sample_y", "pass_rate")
foreach ($col in $required) {
  if (-not ($rows | Select-Object -First 1).PSObject.Properties.Name.Contains($col)) {
    throw "Missing required column '$col' in $Csv"
  }
}

if ($Env.Trim().Length -gt 0) {
  $rows = @($rows | Where-Object { $_.env -eq $Env })
}
if ($rows.Count -eq 0) {
  throw "No rows to plot after filtering."
}

$envOrder = @("sokoban", "frozen_lake")
$envs = @($rows | Group-Object env | ForEach-Object { $_.Name })
$envs = @($envOrder | Where-Object { $envs -contains $_ }) + @($envs | Where-Object { $envOrder -notcontains $_ } | Sort-Object)

$marginLeft = 78
$marginRight = 36
$marginTop = 58
$marginBottom = 68
$innerWidth = $Width - $marginLeft - $marginRight
$panelGap = 34
$Height = $marginTop + $marginBottom + ($PanelHeight * $envs.Count) + ($panelGap * [math]::Max(0, $envs.Count - 1))
$innerHeight = $PanelHeight - 74
$bandwidth = 0.075

$svg = New-Object System.Collections.Generic.List[string]
$svg.Add('<?xml version="1.0" encoding="UTF-8"?>')
$svg.Add("<svg xmlns=`"http://www.w3.org/2000/svg`" width=`"$Width`" height=`"$Height`" viewBox=`"0 0 $Width $Height`">")
$svg.Add("<rect width=`"100%`" height=`"100%`" fill=`"#ffffff`"/>")
$svg.Add("<style>")
$svg.Add("text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#d1d5db;stroke-width:1;stroke-dasharray:3 4}.violin{fill:#7aa6c2;fill-opacity:.42;stroke:#335f78;stroke-width:1}.mean{fill:none;stroke:#d94f45;stroke-width:2.2}.median{stroke:#17445e;stroke-width:1.4}.dot{fill:#143d59;fill-opacity:.72}.tick{font-size:11px}.label{font-size:13px}.panel-title{font-size:15px;font-weight:700}.title{font-size:18px;font-weight:700}.note{font-size:12px;fill:#4b5563}")
$svg.Add("</style>")
$svg.Add("<text x=`"$(ToFixed ($Width / 2.0) 1)`" y=`"28`" text-anchor=`"middle`" class=`"title`">$(SvgEscape $Title)</text>")

for ($panelIndex = 0; $panelIndex -lt $envs.Count; $panelIndex++) {
  $envName = $envs[$panelIndex]
  $panelTop = $marginTop + $panelIndex * ($PanelHeight + $panelGap)
  $plotTop = $panelTop + 34
  $plotBottom = $plotTop + $innerHeight
  $plotLeft = $marginLeft
  $plotRight = $Width - $marginRight

  $envRows = @($rows | Where-Object { $_.env -eq $envName })
  $groups = @($envRows | Group-Object state_x | Sort-Object { [int]$_.Name })
  $stateCount = $groups.Count
  $xStep = if ($stateCount -gt 1) { $innerWidth / ($stateCount - 1) } else { $innerWidth }
  $maxHalfWidth = [math]::Min(13.0, $xStep * 0.38)

  $svg.Add("<text x=`"$plotLeft`" y=`"$($panelTop + 18)`" class=`"panel-title`">$(SvgEscape (EnvLabel $envName))</text>")
  $svg.Add("<text x=`"$plotRight`" y=`"$($panelTop + 18)`" text-anchor=`"end`" class=`"note`">violin = 8 sample pass rates per state; red line = mean</text>")

  foreach ($tick in 0, 0.25, 0.5, 0.75, 1.0) {
    $y = $plotBottom - ([double]$tick * $innerHeight)
    $svg.Add("<line x1=`"$plotLeft`" y1=`"$(ToFixed $y 1)`" x2=`"$plotRight`" y2=`"$(ToFixed $y 1)`" class=`"grid`"/>")
    $svg.Add("<text x=`"$($plotLeft - 10)`" y=`"$(ToFixed ($y + 4) 1)`" text-anchor=`"end`" class=`"tick`">$(ToFixed $tick 2)</text>")
  }

  $svg.Add("<line x1=`"$plotLeft`" y1=`"$plotTop`" x2=`"$plotLeft`" y2=`"$plotBottom`" class=`"axis`"/>")
  $svg.Add("<line x1=`"$plotLeft`" y1=`"$plotBottom`" x2=`"$plotRight`" y2=`"$plotBottom`" class=`"axis`"/>")
  $svg.Add("<text x=`"24`" y=`"$(ToFixed (($plotTop + $plotBottom) / 2.0) 1)`" transform=`"rotate(-90 24 $(ToFixed (($plotTop + $plotBottom) / 2.0) 1))`" text-anchor=`"middle`" class=`"label`">sample accuracy</text>")

  $meanPoints = New-Object System.Collections.Generic.List[string]

  for ($i = 0; $i -lt $groups.Count; $i++) {
    $group = $groups[$i]
    $state = [int]$group.Name
    $values = @($group.Group | ForEach-Object { [double]$_.pass_rate })
    $x = $plotLeft + $i * $xStep

    $densities = New-Object System.Collections.Generic.List[object]
    $maxDensity = 0.0
    for ($j = 0; $j -le 100; $j++) {
      $yv = $j / 100.0
      $d = KernelDensity $values $yv $bandwidth
      if ($d -gt $maxDensity) { $maxDensity = $d }
      $densities.Add([pscustomobject]@{ Y = $yv; D = $d })
    }
    if ($maxDensity -le 0) { $maxDensity = 1.0 }

    $leftPts = New-Object System.Collections.Generic.List[string]
    $rightPts = New-Object System.Collections.Generic.List[string]
    foreach ($item in $densities) {
      $yv = [double]$item.Y
      $half = ([double]$item.D / $maxDensity) * $maxHalfWidth
      $py = $plotBottom - $yv * $innerHeight
      $leftPts.Add("$(ToFixed ($x - $half) 2),$(ToFixed $py 2)")
      $rightPts.Insert(0, "$(ToFixed ($x + $half) 2),$(ToFixed $py 2)")
    }
    $points = ($leftPts + $rightPts) -join " "
    $svg.Add("<polygon points=`"$points`" class=`"violin`"/>")

    $median = Quantile $values 0.5
    $my = $plotBottom - $median * $innerHeight
    $svg.Add("<line x1=`"$(ToFixed ($x - $maxHalfWidth * 0.7) 1)`" y1=`"$(ToFixed $my 1)`" x2=`"$(ToFixed ($x + $maxHalfWidth * 0.7) 1)`" y2=`"$(ToFixed $my 1)`" class=`"median`"/>")

    for ($vIndex = 0; $vIndex -lt $values.Count; $vIndex++) {
      $v = [double]$values[$vIndex]
      $jitterCycle = ($vIndex % 5) - 2
      $jx = $x + $jitterCycle * [math]::Min(2.4, $maxHalfWidth / 5.0)
      $jy = $plotBottom - $v * $innerHeight
      $svg.Add("<circle cx=`"$(ToFixed $jx 2)`" cy=`"$(ToFixed $jy 2)`" r=`"2.1`" class=`"dot`"/>")
    }

    $mean = Mean $values
    $meanY = $plotBottom - $mean * $innerHeight
    $meanPoints.Add("$(ToFixed $x 2),$(ToFixed $meanY 2)")

    if (($state -eq 1) -or ($state % 5 -eq 0) -or ($state -eq [int]$groups[-1].Name)) {
      $svg.Add("<line x1=`"$(ToFixed $x 1)`" y1=`"$plotBottom`" x2=`"$(ToFixed $x 1)`" y2=`"$($plotBottom + 5)`" class=`"axis`"/>")
      $svg.Add("<text x=`"$(ToFixed $x 1)`" y=`"$($plotBottom + 20)`" text-anchor=`"middle`" class=`"tick`">$state</text>")
    }
  }

  $svg.Add("<polyline points=`"$($meanPoints -join ' ')`" class=`"mean`"/>")
  $svg.Add("<text x=`"$(ToFixed (($plotLeft + $plotRight) / 2.0) 1)`" y=`"$($plotBottom + 46)`" text-anchor=`"middle`" class=`"label`">state_x / k</text>")
}

$svg.Add("</svg>")

$outPath = [System.IO.Path]::GetFullPath($Out)
$outDir = [System.IO.Path]::GetDirectoryName($outPath)
if ($outDir -and -not (Test-Path -LiteralPath $outDir)) {
  New-Item -ItemType Directory -Force -Path $outDir | Out-Null
}
$svg -join "`n" | Set-Content -LiteralPath $outPath -Encoding UTF8
Write-Output "Saved plot to $outPath"
