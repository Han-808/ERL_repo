$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$OutputDir = Join-Path $Root "online_notebook_variant_accuracy_plots"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$Methods = @(
    @{
        Key = "notebook_minimal"
        Label = "Original minimal"
        Color = "#111827"
    },
    @{
        Key = "notebook_minimal_thinkahead"
        Label = "Think-ahead"
        Color = "#2563eb"
    },
    @{
        Key = "notebook_minimal_mechanism"
        Label = "Mechanism"
        Color = "#dc2626"
    }
)

$Envs = @(
    @{
        Key = "frozen_lake"
        Label = "FrozenLake"
    },
    @{
        Key = "sokoban"
        Label = "Sokoban"
    }
)

function Get-LatestResultPath {
    param(
        [string]$Method,
        [string]$Env
    )

    $name = "results_${Method}_${Env}.json"
    $files = Get-ChildItem -Path $Root -Recurse -File -Filter $name |
        Sort-Object LastWriteTime -Descending
    if ($files.Count -eq 0) {
        return $null
    }
    return $files[0].FullName
}

function Get-RunningAccuracy {
    param([object[]]$Logs)

    $out = New-Object System.Collections.Generic.List[double]
    $hits = 0
    for ($i = 0; $i -lt $Logs.Count; $i++) {
        $reward = [double]$Logs[$i].reward1
        if ($reward -ge 1.0) {
            $hits += 1
        }
        $out.Add($hits / ($i + 1))
    }
    return $out.ToArray()
}

function Escape-Xml {
    param([string]$Text)
    return [System.Security.SecurityElement]::Escape($Text)
}

function Points-ToString {
    param(
        [double[]]$Values,
        [double]$X0,
        [double]$Y0,
        [double]$PlotW,
        [double]$PlotH,
        [int]$MaxEpisode
    )

    $pts = New-Object System.Collections.Generic.List[string]
    for ($i = 0; $i -lt $Values.Count; $i++) {
        $episode = $i + 1
        $x = if ($MaxEpisode -gt 1) {
            $X0 + (($episode - 1) / ($MaxEpisode - 1)) * $PlotW
        } else {
            $X0
        }
        $y = $Y0 + (1.0 - $Values[$i]) * $PlotH
        $pts.Add(("{0:F2},{1:F2}" -f $x, $y))
    }
    return ($pts -join " ")
}

function Write-LineChart {
    param(
        [string]$Title,
        [string]$Subtitle,
        [array]$Series,
        [string]$OutPath
    )

    $width = 1040
    $height = 640
    $left = 86
    $right = 32
    $top = 86
    $bottom = 90
    $plotW = $width - $left - $right
    $plotH = $height - $top - $bottom
    $maxEpisode = ($Series | ForEach-Object { $_.Values.Count } | Measure-Object -Maximum).Maximum
    if (-not $maxEpisode) {
        $maxEpisode = 40
    }

    $svg = New-Object System.Collections.Generic.List[string]
    $svg.Add("<svg xmlns='http://www.w3.org/2000/svg' width='$width' height='$height' viewBox='0 0 $width $height'>")
    $svg.Add("<rect width='100%' height='100%' fill='#ffffff'/>")
    $svg.Add("<text x='$left' y='40' font-family='Arial, sans-serif' font-size='24' font-weight='700' fill='#111827'>$(Escape-Xml $Title)</text>")
    $svg.Add("<text x='$left' y='66' font-family='Arial, sans-serif' font-size='14' fill='#4b5563'>$(Escape-Xml $Subtitle)</text>")

    for ($t = 0; $t -le 10; $t++) {
        $acc = $t / 10.0
        $y = $top + (1.0 - $acc) * $plotH
        $color = if ($t -eq 0) { "#111827" } else { "#e5e7eb" }
        $svg.Add("<line x1='$left' y1='$("{0:F2}" -f $y)' x2='$($left + $plotW)' y2='$("{0:F2}" -f $y)' stroke='$color' stroke-width='1'/>")
        $svg.Add("<text x='$($left - 12)' y='$("{0:F2}" -f ($y + 4))' text-anchor='end' font-family='Arial, sans-serif' font-size='12' fill='#4b5563'>$($t * 10)%</text>")
    }

    foreach ($episode in @(1, 10, 20, 30, 40)) {
        if ($episode -gt $maxEpisode) {
            continue
        }
        $x = $left + (($episode - 1) / [Math]::Max(1, $maxEpisode - 1)) * $plotW
        $svg.Add("<line x1='$("{0:F2}" -f $x)' y1='$top' x2='$("{0:F2}" -f $x)' y2='$($top + $plotH)' stroke='#f3f4f6' stroke-width='1'/>")
        $svg.Add("<text x='$("{0:F2}" -f $x)' y='$($top + $plotH + 28)' text-anchor='middle' font-family='Arial, sans-serif' font-size='12' fill='#4b5563'>$episode</text>")
    }

    $svg.Add("<line x1='$left' y1='$top' x2='$left' y2='$($top + $plotH)' stroke='#111827' stroke-width='1.3'/>")
    $svg.Add("<line x1='$left' y1='$($top + $plotH)' x2='$($left + $plotW)' y2='$($top + $plotH)' stroke='#111827' stroke-width='1.3'/>")
    $svg.Add("<text x='$($left + $plotW / 2)' y='$($height - 28)' text-anchor='middle' font-family='Arial, sans-serif' font-size='15' fill='#111827'>Episode</text>")
    $svg.Add("<text transform='translate(24 $($top + $plotH / 2)) rotate(-90)' text-anchor='middle' font-family='Arial, sans-serif' font-size='15' fill='#111827'>Running accuracy</text>")

    $legendX = $left + 12
    $legendY = $top + 16
    for ($i = 0; $i -lt $Series.Count; $i++) {
        $s = $Series[$i]
        $y = $legendY + $i * 24
        $svg.Add("<line x1='$legendX' y1='$y' x2='$($legendX + 28)' y2='$y' stroke='$($s.Color)' stroke-width='4'/>")
        $svg.Add("<text x='$($legendX + 38)' y='$($y + 5)' font-family='Arial, sans-serif' font-size='13' fill='#111827'>$(Escape-Xml $s.Label)</text>")
    }

    foreach ($s in $Series) {
        $points = Points-ToString -Values $s.Values -X0 $left -Y0 $top -PlotW $plotW -PlotH $plotH -MaxEpisode $maxEpisode
        $svg.Add("<polyline points='$points' fill='none' stroke='$($s.Color)' stroke-width='3.2' stroke-linejoin='round' stroke-linecap='round'/>")
        for ($i = 0; $i -lt $s.Values.Count; $i += 4) {
            $episode = $i + 1
            $x = $left + (($episode - 1) / [Math]::Max(1, $maxEpisode - 1)) * $plotW
            $y = $top + (1.0 - $s.Values[$i]) * $plotH
            $svg.Add("<circle cx='$("{0:F2}" -f $x)' cy='$("{0:F2}" -f $y)' r='3.3' fill='#ffffff' stroke='$($s.Color)' stroke-width='2'/>")
        }
    }

    $svg.Add("</svg>")
    Set-Content -Path $OutPath -Value ($svg -join "`n") -Encoding UTF8
}

$records = New-Object System.Collections.Generic.List[object]
$seriesByEnv = @{}

foreach ($env in $Envs) {
    $envKey = [string]$env["Key"]
    $seriesByEnv[$envKey] = New-Object System.Collections.Generic.List[object]
    foreach ($method in $Methods) {
        $methodKey = [string]$method["Key"]
        $methodLabel = [string]$method["Label"]
        $methodColor = [string]$method["Color"]
        $path = Get-LatestResultPath -Method $methodKey -Env $envKey
        if (-not $path) {
            $records.Add([pscustomobject]@{
                method = $methodKey
                env = $envKey
                status = "missing"
                episodes = 0
                success = 0
                pass_rate = ""
                first20 = ""
                last20 = ""
                result_path = ""
            })
            continue
        }

        $data = Get-Content -Raw -LiteralPath $path | ConvertFrom-Json
        $logs = @($data.logs)
        $values = Get-RunningAccuracy -Logs $logs
        $success = @($logs | Where-Object { [double]$_.reward1 -ge 1.0 }).Count
        $first20 = @($logs | Select-Object -First 20)
        $last20 = @($logs | Select-Object -Last 20)
        $firstSuccess = @($first20 | Where-Object { [double]$_.reward1 -ge 1.0 }).Count
        $lastSuccess = @($last20 | Where-Object { [double]$_.reward1 -ge 1.0 }).Count
        $firstRate = if ($first20.Count) { $firstSuccess / $first20.Count } else { 0 }
        $lastRate = if ($last20.Count) { $lastSuccess / $last20.Count } else { 0 }

        $seriesByEnv[$envKey].Add([pscustomobject]@{
            Label = $methodLabel
            Color = $methodColor
            Values = $values
        })

        $records.Add([pscustomobject]@{
            method = $methodKey
            env = $envKey
            status = "ok"
            episodes = $logs.Count
            success = $success
            pass_rate = if ($logs.Count) { $success / $logs.Count } else { 0 }
            first20 = $firstRate
            last20 = $lastRate
            result_path = $path
        })
    }
}

foreach ($env in $Envs) {
    $envKey = [string]$env["Key"]
    $envLabel = [string]$env["Label"]
    $seriesList = $seriesByEnv[$envKey]
    $series = $seriesList.ToArray()
    Write-LineChart `
        -Title "${envLabel}: Online Running Accuracy" `
        -Subtitle "Qwen3-14B no-thinking, 40 online episodes; lines show cumulative success rate" `
        -Series $series `
        -OutPath (Join-Path $OutputDir "${envKey}_online_accuracy.svg")
}

$combinedSeries = New-Object System.Collections.Generic.List[object]
foreach ($method in $Methods) {
    $methodLabel = [string]$method["Label"]
    $methodColor = [string]$method["Color"]
    $fl = @($seriesByEnv["frozen_lake"] | Where-Object { $_.Label -eq $methodLabel })
    $sk = @($seriesByEnv["sokoban"] | Where-Object { $_.Label -eq $methodLabel })
    if ($fl.Count -eq 0 -or $sk.Count -eq 0) {
        continue
    }
    $n = [Math]::Min($fl[0].Values.Count, $sk[0].Values.Count)
    $vals = New-Object System.Collections.Generic.List[double]
    for ($i = 0; $i -lt $n; $i++) {
        $vals.Add(($fl[0].Values[$i] + $sk[0].Values[$i]) / 2.0)
    }
    $combinedSeries.Add([pscustomobject]@{
        Label = $methodLabel
        Color = $methodColor
        Values = $vals.ToArray()
    })
}

Write-LineChart `
    -Title "FrozenLake + Sokoban: Mean Online Running Accuracy" `
    -Subtitle "Mean of the two environment cumulative success rates by episode" `
    -Series $combinedSeries.ToArray() `
    -OutPath (Join-Path $OutputDir "combined_mean_online_accuracy.svg")

$records | Export-Csv -NoTypeInformation -Path (Join-Path $OutputDir "online_accuracy_summary.csv")

Write-Output "Wrote $OutputDir"
Get-ChildItem -Path $OutputDir | Select-Object Name,Length | Format-Table -AutoSize
