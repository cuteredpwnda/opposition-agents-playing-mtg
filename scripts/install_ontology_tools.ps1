param()

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tools = Join-Path $root ".tools"
New-Item -ItemType Directory -Force -Path $tools | Out-Null

function Install-CheckedArchive {
    param(
        [string]$Name, [string]$Url, [string]$Hash,
        [string]$Algorithm, [string]$ArchiveDirectory
    )
    $destination = Join-Path $tools $Name
    if (Test-Path -LiteralPath $destination) {
        Write-Output "Already present: $destination"
        return
    }
    $archive = Join-Path $tools "$Name.zip"
    $staging = Join-Path $tools "$Name-install"
    if (Test-Path -LiteralPath $staging) {
        throw "Installation staging directory already exists: $staging. Inspect it before retrying."
    }
    & curl.exe --fail --location --silent --show-error -o $archive $Url
    if ($LASTEXITCODE -ne 0) { throw "Download failed: $Name" }
    $actual = (Get-FileHash -LiteralPath $archive -Algorithm $Algorithm).Hash.ToLower()
    if ($actual -ne $Hash) { throw "Checksum mismatch: $Name" }
    Expand-Archive -LiteralPath $archive -DestinationPath $staging
    Move-Item -LiteralPath (Join-Path $staging $ArchiveDirectory) -Destination $destination
    Remove-Item -LiteralPath $archive
    Remove-Item -LiteralPath $staging
}

Install-CheckedArchive `
    -Name "java" `
    -Url "https://github.com/adoptium/temurin21-binaries/releases/download/jdk-21.0.12.1%2B1/OpenJDK21U-jdk_x64_windows_hotspot_21.0.12.1_1.zip" `
    -Hash "f9d6e191ab098c0d416e7d588a24420a8621cd2f4720dab2459b8b7b2d2d8b4e" `
    -Algorithm "SHA256" `
    -ArchiveDirectory "jdk-21.0.12.1+1"

Install-CheckedArchive `
    -Name "fuseki" `
    -Url "https://downloads.apache.org/jena/binaries/apache-jena-fuseki-6.2.0.zip" `
    -Hash "46e5d798faf80fe5f4b32318750071b9172315f9d86bb3aa3ba4d5e94abe2e21cd194eab349d491a203c934c6e59b370a671b338f2a413110e859dc628ffe934" `
    -Algorithm "SHA512" `
    -ArchiveDirectory "apache-jena-fuseki-6.2.0"

& (Join-Path $tools "java\bin\java.exe") -version
if ($LASTEXITCODE -ne 0) { throw "Local Java verification failed" }
& (Join-Path $tools "java\bin\java.exe") -jar (Join-Path $tools "fuseki\fuseki-server.jar") --version
if ($LASTEXITCODE -ne 0) { throw "Local Fuseki verification failed" }
