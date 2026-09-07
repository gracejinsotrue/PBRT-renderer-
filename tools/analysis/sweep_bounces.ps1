# Real-time feasibility probe: GPU cost per frame vs path length.
#
# Real-time builds run 1-2 bounces, not the offline default of 32. This sweeps
# MAX_BOUNCES across the coverage scenes and reports pure DispatchRays GPU time
# (headless --profile timestamps, min-of-N), which is the number a frame budget
# is spent against.
#
# Usage:  ./tools/analysis/sweep_bounces.ps1
#         ./tools/analysis/sweep_bounces.ps1 -Bounces 1,2,4 -CoolSeconds 20
param(
    [int[]]$Bounces = @(2, 32),
    [int]$CoolSeconds = 15
)

# NB: deliberately NOT 'Stop'. The renderer writes progress to stderr, and in
# PS 5.1 a native command's stderr surfaces as NativeCommandError - under 'Stop'
# that would abort the sweep on a perfectly healthy run.
$ErrorActionPreference = 'Continue'

# Every path below is repo-root-relative; anchor there rather than trusting the
# caller's working directory.
Set-Location (Resolve-Path (Join-Path $PSScriptRoot '../..'))

$dxc = "C:/VulkanSDK/1.4.341.1/Bin/dxc.exe"
$cso = "build/Release/Shaders.cso"
$exe = "./build/Release/nori-dxr.exe"

# scene label -> path. All are low-spp profile copies with no <integrator> tag.
$scenes = [ordered]@{
    'cbox (2k tris)'     = 'scenes/final_scenes/cbox/cbox_p3.xml'
    'hand-SSS (750k)'    = 'scenes/hand/hand_profile.xml'
    'hair (tess tubes)'  = 'scenes/final_scenes/hair/hair_profile.xml'
    'hero self-portrait' = 'scenes/final_scene_real/scene2_profile.xml'
}

# The sweep overwrites Shaders.cso; stash whatever is there and put it back.
$stash = "build/Release/Shaders.sweepstash.cso"
Copy-Item $cso $stash -Force
Write-Host "[sweep] stashed current shader -> $stash" -ForegroundColor DarkGray

$results = @()
try {
    foreach ($b in $Bounces) {
        Write-Host "`n=== MAX_BOUNCES=$b ===" -ForegroundColor Cyan
        & $dxc -T lib_6_5 -D USE_RAYQUERY=1 -D MAX_BOUNCES=$b -Fo $cso shaders/Shaders.hlsl
        if ($LASTEXITCODE -ne 0) { throw "shader compile failed at MAX_BOUNCES=$b" }

        foreach ($name in $scenes.Keys) {
            $path = $scenes[$name]
            if (-not (Test-Path $path)) {
                Write-Host ("  {0,-20} SKIP (missing {1})" -f $name, $path) -ForegroundColor Yellow
                continue
            }

            # stdout only: the [profile] summary goes to stdout, progress to stderr.
            $out = (& $exe $path --headless --profile) | Out-String

            $m = [regex]::Match($out, 'min\s+([\d.]+)\s+median\s+([\d.]+)\s+mean\s+([\d.]+)\s+p95\s+([\d.]+)')
            if ($m.Success) {
                $min = [double]$m.Groups[1].Value
                $med = [double]$m.Groups[2].Value

                # Headroom verdict against a 16.7 ms budget, reserving ~5 ms for a denoiser.
                if     ($min -le 11.0) { $fits = 'yes' }
                elseif ($min -le 16.7) { $fits = 'tight' }
                else                   { $fits = 'NO' }

                $results += [pscustomobject]@{
                    Bounces  = $b
                    Scene    = $name
                    MinMs    = [math]::Round($min, 2)
                    MedianMs = [math]::Round($med, 2)
                    FpsAtMin = [math]::Round(1000.0 / $min, 1)
                    Fits60   = $fits
                }
                Write-Host ("  {0,-20} min {1,7:N2} ms  median {2,7:N2} ms  ({3} fps)  {4}" -f `
                    $name, $min, $med, [math]::Round(1000.0 / $min, 1), $fits)
            }
            else {
                Write-Host ("  {0,-20} FAILED - no [profile] line in output" -f $name) -ForegroundColor Red
            }

            if ($CoolSeconds -gt 0) { Start-Sleep -Seconds $CoolSeconds }
        }
    }
}
finally {
    Copy-Item $stash $cso -Force
    Remove-Item $stash -Force
    Write-Host "`n[sweep] restored original shader" -ForegroundColor DarkGray
}

Write-Host "`n===== SUMMARY (pure DispatchRays GPU ms) =====" -ForegroundColor Green
Write-Host "Fits60: 'yes' = under 11 ms, leaving ~5 ms for a denoiser at 60 fps`n"
$results | Format-Table Bounces, Scene, MinMs, MedianMs, FpsAtMin, Fits60 -AutoSize
$results | Export-Csv -NoTypeInformation -Path bounce_sweep.csv
Write-Host "[sweep] wrote bounce_sweep.csv"
