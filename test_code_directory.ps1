$directory = "./app"
$endpoint = "http://localhost:8000/api/code/process-directory"

$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"

# Virgülle ayrılmış liste olarak diller
$languages = "py,js,ts,json"

$bodyLines = @(
    "--$boundary",
    "Content-Disposition: form-data; name=`"directory`"$LF",
    $directory,
    "--$boundary",
    "Content-Disposition: form-data; name=`"languages`"$LF",
    $languages,
    "--$boundary--$LF"
)

$body = $bodyLines -join $LF

try {
    Write-Host "Sending request to process directory: $directory with languages: $languages"
    $response = Invoke-WebRequest -Uri $endpoint -Method Post -ContentType "multipart/form-data; boundary=$boundary" -Body $body
    Write-Host "Status Code: $($response.StatusCode)"
    Write-Host "Response: $($response.Content)"
    
    # Parse JSON response
    $responseObj = $response.Content | ConvertFrom-Json
    Write-Host "`nStatus: $($responseObj.status)"
    Write-Host "Message: $($responseObj.message)"
    Write-Host "Directory: $($responseObj.directory)"
    Write-Host "Supported Languages: $($responseObj.supported_languages -join ', ')"
} catch {
    Write-Host "Error: $_"
    Write-Host "Status Code: $($_.Exception.Response.StatusCode.value__)"
    if ($_.Exception.Response) {
        $reader = New-Object System.IO.StreamReader($_.Exception.Response.GetResponseStream())
        $reader.BaseStream.Position = 0
        $reader.DiscardBufferedData()
        $responseBody = $reader.ReadToEnd()
        Write-Host "Response body: $responseBody"
    }
} 