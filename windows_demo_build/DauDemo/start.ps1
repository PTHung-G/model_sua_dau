$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Test-CompatiblePython([string]$Exe, [string[]]$PrefixArgs = @()) {
    try {
        & $Exe @PrefixArgs -c 'import sys; sys.exit(not ((3,10) <= sys.version_info[:2] <= (3,12) and sys.maxsize > 2**32))' *> $null
        return ($LASTEXITCODE -eq 0)
    } catch { return $false }
}
try {
    if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows x64 is required.' }
    $python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path -LiteralPath $python)) {
        $basePython = $null
        if (Get-Command py -ErrorAction SilentlyContinue) {
            foreach ($version in @('-3.12', '-3.11', '-3.10')) {
                if (Test-CompatiblePython 'py' @($version)) {
                    $basePython = (& py $version -c 'import sys; print(sys.executable)' | Select-Object -Last 1).Trim()
                    break
                }
            }
        }
        if (-not $basePython -and (Get-Command python -ErrorAction SilentlyContinue)) {
            if (Test-CompatiblePython 'python') {
                $basePython = (& python -c 'import sys; print(sys.executable)' | Select-Object -Last 1).Trim()
            }
        }
        $privatePython = Join-Path $PSScriptRoot 'runtime\python.exe'
        if (-not $basePython -and (Test-Path -LiteralPath $privatePython)) {
            if (Test-CompatiblePython $privatePython) { $basePython = $privatePython }
        }
        if (-not $basePython) { throw 'No compatible Python found. Install Python 3.10-3.12 x64, then run again. This launcher does NOT download Python.' }
        Write-Host "Using existing Python: $basePython"
        & $basePython -m venv (Join-Path $PSScriptRoot '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'Failed to create virtual environment.' }
    }
    if (-not (Test-CompatiblePython $python)) { throw 'Existing .venv is incompatible. Rename that folder and run again.' }
    $check = Join-Path $PSScriptRoot 'check_dependencies.py'
    & $python $check *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'Installing CPU libraries into .venv. Internet required for first setup.'
        & $python -m pip install --disable-pip-version-check 'numpy==2.2.6'
        if ($LASTEXITCODE -ne 0) { throw 'NumPy installation failed.' }
        & $python -m pip install --disable-pip-version-check 'torch==2.6.0' --index-url https://download.pytorch.org/whl/cpu
        if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
        & $python $check
        if ($LASTEXITCODE -ne 0) { throw 'Dependency import failed. If a DLL is missing, install Microsoft Visual C++ Redistributable x64.' }
    }
    & $python (Join-Path $PSScriptRoot 'server.py')
    if ($LASTEXITCODE -ne 0) { throw "Demo exited with code $LASTEXITCODE" }
} catch {
    Write-Host $_ -ForegroundColor Red
    exit 1
}
