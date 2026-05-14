param(
  [Parameter(Mandatory = $true)]
  [string]$SampleCsv,

  [Parameter(Mandatory = $true)]
  [string]$Out,

  [string]$Title = "Median and Std Accuracy by State",

  [string]$Env = "",

  [int]$Width = 1320,

  [int]$PanelHeight = 330
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

function ToFixed([double]$Value, [int]$Digits = 3) {
  return $Value.ToString("F$Digits", [Globalization.CultureInfo]::InvariantCulture)
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

function StdDevSample($Values) {
  $arr = @($Values | ForEach-Object { [double]$_ })
  if ($arr.Count -le 1) { return 0.0 }
  $mean = ($arr | Measure-Object -Average).Average
  $sumSq = 0.0
  foreach ($v in $arr) {
    $sumSq += [math]::Pow($v - $mean, 2)
  }
  return [math]::Sqrt($sumSq / ($arr.Count - 1))
}

if (-not (Test-Path -LiteralPath $SampleCsv)) {
  throw "CSV not found: $SampleCsv"
}

$rows = @(Import-Csv -LiteralPath $SampleCsv)
$required = @("env", "state_x", "pass_rate")
foreach ($col in $required) {
  if (-not ($rows | Select-Object -First 1).PSObject.Properties.Name.Contains($col)) {
    throw "Missing required column '$col' in $SampleCsv"
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
$marginRight = 38
$marginTop = 58
$marginBottom = 64
$panelGap = 34
$innerWidth = $Width - $marginLeft - $marginRight
$Height = $marginTop + $marginBottom + ($PanelHeight * $envs.Count) + ($panelGap * [math]::Max(0, $envs.Count - 1))
$innerHeight = $PanelHeight - 72

$svg = New-Object System.Collections.Generic.List[string]
$svg.Add('<?xml version="1.0" encoding="UTF-8"?>')
$svg.Add("<svg xmlns=`"http://www.w3.org/2000/svg`" width=`"$Width`" height=`"$Height`" viewBox=`"0 0 $Width $Height`">")
$svg.Add("<rect width=`"100%`" height=`"100%`" fill=`"#ffffff`"/>")
$svg.Add("<style>")
$svg.Add("text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}.axis{stroke:#374151;stroke-width:1}.grid{stroke:#d1d5db;stroke-width:1;stroke-dasharray:3 4}.median-line{fill:none;stroke:#1f77b4;stroke-width:2.4}.std-line{fill:none;stroke:#d62728;stroke-width:2.4}.median-dot{fill:#1f77b4}.std-dot{fill:#d62728}.tick{font-size:11px}.label{font-size:13px}.panel-title{font-size:15px;font-weight:700}.title{font-size:18px;font-weight:700}.note{font-size:12px;fill:#4b5563}.legend{font-size:12px}")
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

  $svg.Add("<text x=`"$plotLeft`" y=`"$($panelTop + 18)`" class=`"panel-title`">$(SvgEscape (EnvLabel $envName))</text>")
  $svg.Add("<line x1=`"$(ToFixed ($plotRight - 205) 1)`" y1=`"$($panelTop + 13)`" x2=`"$(ToFixed ($plotRight - 170) 1)`" y2=`"$($panelTop + 13)`" class=`"median-line`"/>")
  $svg.Add("<text x=`"$(ToFixed ($plotRight - 164) 1)`" y=`"$($panelTop + 17)`" class=`"legend`">median</text>")
  $svg.Add("<line x1=`"$(ToFixed ($plotRight - 105) 1)`" y1=`"$($panelTop + 13)`" x2=`"$(ToFixed ($plotRight - 70) 1)`" y2=`"$($panelTop + 13)`" class=`"std-line`"/>")
  $svg.Add("<text x=`"$(ToFixed ($plotRight - 64) 1)`" y=`"$($panelTop + 17)`" class=`"legend`">std</text>")

  foreach ($tick in 0, 0.25, 0.5, 0.75, 1.0) {
    $y = $plotBottom - ([double]$tick * $innerHeight)
    $svg.Add("<line x1=`"$plotLeft`" y1=`"$(ToFixed $y 1)`" x2=`"$plotRight`" y2=`"$(ToFixed $y 1)`" class=`"grid`"/>")
    $svg.Add("<text x=`"$($plotLeft - 10)`" y=`"$(ToFixed ($y + 4) 1)`" text-anchor=`"end`" class=`"tick`">$(ToFixed $tick 2)</text>")
  }

  $svg.Add("<line x1=`"$plotLeft`" y1=`"$plotTop`" x2=`"$plotLeft`" y2=`"$plotBottom`" class=`"axis`"/>")
  $svg.Add("<line x1=`"$plotLeft`" y1=`"$plotBottom`" x2=`"$plotRight`" y2=`"$plotBottom`" class=`"axis`"/>")
  $svg.Add("<text x=`"24`" y=`"$(ToFixed (($plotTop + $plotBottom) / 2.0) 1)`" transform=`"rotate(-90 24 $(ToFixed (($plotTop + $plotBottom) / 2.0) 1))`" text-anchor=`"middle`" class=`"label`">accuracy</text>")

  $medianPoints = New-Object System.Collections.Generic.List[string]
  $stdPoints = New-Object System.Collections.Generic.List[string]

  for ($i = 0; $i -lt $groups.Count; $i++) {
    $group = $groups[$i]
    $state = [int]$group.Name
    $values = @($group.Group | ForEach-Object { [double]$_.pass_rate })
    $median = Quantile $values 0.5
    $std = StdDevSample $values
    $x = $plotLeft + $i * $xStep
    $medianY = $plotBottom - $median * $innerHeight
    $stdY = $plotBottom - $std * $innerHeight
    $medianPoints.Add("$(ToFixed $x 2),$(ToFixed $medianY 2)")
    $stdPoints.Add("$(ToFixed $x 2),$(ToFixed $stdY 2)")
    $svg.Add("<circle cx=`"$(ToFixed $x 2)`" cy=`"$(ToFixed $medianY 2)`" r=`"2.4`" class=`"median-dot`"/>")
    $svg.Add("<circle cx=`"$(ToFixed $x 2)`" cy=`"$(ToFixed $stdY 2)`" r=`"2.4`" class=`"std-dot`"/>")

    if (($state -eq 1) -or ($state % 5 -eq 0) -or ($state -eq [int]$groups[-1].Name)) {
      $svg.Add("<line x1=`"$(ToFixed $x 1)`" y1=`"$plotBottom`" x2=`"$(ToFixed $x 1)`" y2=`"$($plotBottom + 5)`" class=`"axis`"/>")
      $svg.Add("<text x=`"$(ToFixed $x 1)`" y=`"$($plotBottom + 20)`" text-anchor=`"middle`" class=`"tick`">$state</text>")
    }
  }

  $svg.Add("<polyline points=`"$($medianPoints -join ' ')`" class=`"median-line`"/>")
  $svg.Add("<polyline points=`"$($stdPoints -join ' ')`" class=`"std-line`"/>")
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
