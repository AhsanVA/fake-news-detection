$sourcePath = "C:\Users\adhil\Documents\TCS - Internship\fake_news_detection"
$destinationTuple = "C:\Users\adhil\Documents\TCS - Internship\Fake_News_Deliverables.zip"

Write-Host "Zipping project files..."

# List of items to include
$include = @(
    "src",
    "app",
    "docs",
    "models",
    "notebooks",
    "data",
    "requirements.txt",
    "README.md",
    "main.py"
)

# Temporary folder for clean zipping
$tempDir = Join-Path $env:TEMP "fake_news_temp"
if (Test-Path $tempDir) { Remove-Item -Recurse -Force $tempDir }
New-Item -ItemType Directory -Path $tempDir | Out-Null

foreach ($item in $include) {
    $src = Join-Path $sourcePath $item
    if (Test-Path $src) {
        Copy-Item -Recurse -Path $src -Destination $tempDir
    }
}

# Zip
if (Test-Path $destinationTuple) { Remove-Item -Force $destinationTuple }
Compress-Archive -Path "$tempDir\*" -DestinationPath $destinationTuple

# Cleanup
Remove-Item -Recurse -Force $tempDir

Write-Host "Done. Zip created at: $destinationTuple"
