[CmdletBinding()]
param(
    [string]$BuildDirectory = 'target/release',
    [string]$OutputDirectory = 'target/cabi-release'
)

$ErrorActionPreference = 'Stop'
Push-Location (Split-Path -Parent $PSScriptRoot)
try {
    $metadata = cargo metadata --locked --no-deps --format-version 1
    if ($LASTEXITCODE -ne 0) { throw 'Could not read Cargo metadata.' }
    $version = ($metadata | ConvertFrom-Json).packages.Where({ $_.name -eq 'umap-capi' }).version
    $build = (Resolve-Path -LiteralPath $BuildDirectory).Path
    $out = (New-Item -ItemType Directory -Force -Path $OutputDirectory).FullName
    $staging = Join-Path 'target/cabi-verification' ([guid]::NewGuid().ToString('N'))
    $package = (New-Item -ItemType Directory -Path (Join-Path $staging 'package') -Force).FullName
    $archiveName = "umap-alternative-cabi-$version-win-x64.zip"
    $archive = Join-Path (Split-Path -Parent $package) $archiveName

    Copy-Item -LiteralPath (Join-Path $build 'umap_capi.dll') -Destination $package
    Copy-Item -LiteralPath (Join-Path $build 'umap_capi.dll.lib') -Destination (Join-Path $package 'umap_capi.lib')
    Copy-Item -LiteralPath 'crates/umap-capi/include/umap_capi.h' -Destination $package
    Copy-Item -LiteralPath 'crates/umap-capi/examples/fit_graph.c' -Destination $package
    Copy-Item -LiteralPath 'crates/umap-capi/README.md' -Destination $package
    Copy-Item -LiteralPath 'LICENSE' -Destination $package
    Compress-Archive -LiteralPath (Get-ChildItem -LiteralPath $package -File).FullName -DestinationPath $archive

    $extracted = (New-Item -ItemType Directory -Path (Join-Path $staging 'extracted')).FullName
    Expand-Archive -LiteralPath $archive -DestinationPath $extracted

    $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio/Installer/vswhere.exe'
    $vs = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if (-not $vs) { throw 'Visual Studio with the C++ toolchain was not found.' }
    Import-Module (Join-Path $vs 'Common7/Tools/Microsoft.VisualStudio.DevShell.dll')
    Enter-VsDevShell -VsInstallPath $vs -SkipAutomaticLocation -DevCmdArguments '-arch=x64 -host_arch=x64'

    $sources = @(
        'crates/umap-capi/tests/c_smoke.c'
        'crates/umap-capi/tests/c_inverse_smoke.c'
        (Join-Path $extracted 'fit_graph.c')
    )
    foreach ($source in $sources) {
        $name = [IO.Path]::GetFileNameWithoutExtension($source)
        $exe = Join-Path $extracted "$name.exe"
        & cl.exe /nologo /std:c11 /W4 /WX "/I$extracted" $source `
            "/Fo$(Join-Path $extracted "$name.obj")" "/Fe$exe" (Join-Path $extracted 'umap_capi.lib')
        if ($LASTEXITCODE -ne 0) { throw "Compiling $source failed ($LASTEXITCODE)." }
        & $exe
        if ($LASTEXITCODE -ne 0) { throw "Running $name failed ($LASTEXITCODE)." }
    }

    Copy-Item -LiteralPath $archive -Destination $out
    Write-Output (Join-Path $out $archiveName)
} finally {
    Pop-Location
}
