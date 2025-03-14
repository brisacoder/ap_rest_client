# PowerShell 7 script to add dependencies from requirements.txt to poetry

# Path to your requirements.txt file
$requirementsPath = "requirements.txt"

# Check if requirements.txt exists
if (!(Test-Path $requirementsPath)) {
    Write-Error "File $requirementsPath not found."
    exit 1
}

# Read each line, ignoring comments and empty lines
Get-Content $requirementsPath | ForEach-Object {
    $line = $_.Trim()

    # Skip empty lines and comments
    if (![string]::IsNullOrWhiteSpace($line) -and !$line.StartsWith('#')) {
        Write-Host "Adding dependency: $line"
        poetry add $line
    }
}
