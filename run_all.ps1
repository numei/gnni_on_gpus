[CmdletBinding()]
param(
    [string]$Root = $PSScriptRoot,
    [switch]$SkipCuda,
    [switch]$SkipTests,
    [switch]$SkipDataset,
    [string]$CudaArchitecture
)

$ErrorActionPreference = 'Stop'
Set-Location $Root

function Invoke-Checked {
    param(
        [string]$Command,
        [string[]]$Arguments
    )

    Write-Host "`n> $Command $($Arguments -join ' ')" -ForegroundColor Cyan
    & $Command @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed (exit code $LASTEXITCODE): $Command"
    }
}

function Find-CommandPath {
    param([string]$Name)
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) {
        return $command.Source
    }
    return $null
}

function Find-NvccPath {
    $candidates = @()
    $pathNvcc = Find-CommandPath 'nvcc.exe'
    if ($pathNvcc) {
        $candidates += $pathNvcc
    }

    if ($env:CUDA_PATH) {
        $candidates += (Join-Path $env:CUDA_PATH 'bin\nvcc.exe')
    }

    $registryPaths = @(
        'HKLM:\SOFTWARE\NVIDIA Corporation\GPU Computing Toolkit\CUDA',
        'HKLM:\SOFTWARE\WOW6432Node\NVIDIA Corporation\GPU Computing Toolkit\CUDA'
    )
    foreach ($registryPath in $registryPaths) {
        $cudaKey = Get-ItemProperty -Path $registryPath -ErrorAction SilentlyContinue
        if ($cudaKey -and $cudaKey.CurrentVersion) {
            $candidates += Join-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\$($cudaKey.CurrentVersion)" 'bin\nvcc.exe'
        }
    }

    $cudaRoot = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if (Test-Path $cudaRoot) {
        $candidates += Get-ChildItem -Path $cudaRoot -Directory -ErrorAction SilentlyContinue |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName 'bin\nvcc.exe' }
    }

    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }
    return $null
}

function Initialize-MsvcEnvironment {
    if (Find-CommandPath 'cl.exe') {
        return
    }

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
    if (-not (Test-Path $vswhere)) {
        throw 'Visual Studio Build Tools not found. Install MSVC C++, Windows SDK, and CMake tools.'
    }

    $installationPath = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $installationPath) {
        throw 'No Visual Studio installation with the MSVC C++ toolset was found.'
    }

    $vsDevCmd = Join-Path $installationPath 'Common7\Tools\VsDevCmd.bat'
    if (-not (Test-Path $vsDevCmd)) {
        throw "Visual Studio developer script not found: $vsDevCmd"
    }

    $envDump = & $env:ComSpec /s /c "`"$vsDevCmd`" -arch=x64 > `"$env:TEMP\gnni-vs-env.txt`" && set"
    foreach ($line in $envDump) {
        if ($line -match '^([^=]+)=(.*)$') {
            [Environment]::SetEnvironmentVariable($Matches[1], $Matches[2], 'Process')
        }
    }

    if (-not (Find-CommandPath 'cl.exe')) {
        throw 'Failed to load the Visual Studio environment; cl.exe is still unavailable.'
    }
}

function Get-CmakeGeneratorArguments {
    $ninja = Find-CommandPath 'ninja.exe'
    if ($ninja) {
        return @('-G', 'Ninja')
    }
    if ($env:VisualStudioVersion -like '16.*') {
        return @('-G', 'Visual Studio 16 2019', '-A', 'x64')
    }
    return @('-G', 'Visual Studio 17 2022', '-A', 'x64')
}

Initialize-MsvcEnvironment

$cmake = Find-CommandPath 'cmake.exe'
if (-not $cmake) {
    throw 'cmake.exe not found. Install CMake and add it to PATH.'
}

$generatorArguments = Get-CmakeGeneratorArguments
$commonConfigure = @('-S', '.', '-DCMAKE_BUILD_TYPE=Release') + $generatorArguments

Write-Host "C++ compiler: $((Find-CommandPath 'cl.exe'))"

Invoke-Checked $cmake (@('-B', 'build-cpu', '-DGNNI_ENABLE_CUDA=OFF') + $commonConfigure)
Invoke-Checked $cmake @('--build', 'build-cpu', '--config', 'Release', '--parallel')

if (-not $SkipTests) {
    Invoke-Checked 'ctest' @('--test-dir', 'build-cpu', '-C', 'Release', '--output-on-failure')
}

if (-not $SkipDataset -and -not (Test-Path '.\graph.gnni')) {
    Invoke-Checked '.\build-cpu\common\gnni_dataset_generator.exe' @(
        'ba', '--vertices', '10000', '--edges-per-new-vertex', '8',
        '--features', '64', '--seed', '42', '--self-loops',
        '--output', 'graph.gnni'
    )
} elseif (-not (Test-Path '.\graph.gnni')) {
    throw 'graph.gnni does not exist. Remove -SkipDataset to generate it.'
} else {
    Write-Host "`nUsing existing graph.gnni."
}

Invoke-Checked '.\build-cpu\gnni_q1.exe' @(
    '--dataset', 'graph.gnni', '--layers', '64,32',
    '--backend', 'cpu', '--verify'
)

if (-not $SkipCuda) {
    $nvcc = Find-NvccPath
    if (-not $nvcc) {
        throw 'nvcc.exe was not found. Add CUDA to PATH or set CUDA_PATH, or use -SkipCuda for CPU only.'
    }
    Write-Host "CUDA compiler: $nvcc"

    $cudaConfigure = @('-B', 'build-gpu', '-DGNNI_ENABLE_CUDA=ON',
        "-DCMAKE_CUDA_COMPILER=$nvcc")
    if ($CudaArchitecture) {
        $cudaConfigure += "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitecture"
    }
    Invoke-Checked $cmake ($cudaConfigure + $commonConfigure)
    Invoke-Checked $cmake @('--build', 'build-gpu', '--config', 'Release', '--parallel')

    if (-not $SkipTests) {
        Invoke-Checked 'ctest' @('--test-dir', 'build-gpu', '-C', 'Release', '--output-on-failure')
    }

    Invoke-Checked '.\build-gpu\gnni_q1.exe' @(
        '--dataset', 'graph.gnni', '--layers', '64,32',
        '--backend', 'cuda-tiled', '--verify'
    )
}

Write-Host "`nCPU/CUDA project run completed." -ForegroundColor Green
