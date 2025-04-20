$filePath = "test_upload.py"
$language = "python"
$endpoint = "http://localhost:8000/api/code/upload"

$fileBytes = [System.IO.File]::ReadAllBytes($filePath)
$fileName = [System.IO.Path]::GetFileName($filePath)

$boundary = [System.Guid]::NewGuid().ToString()
$LF = "`r`n"

$bodyLines = @(
    "--$boundary",
    "Content-Disposition: form-data; name=`"file`"; filename=`"$fileName`"",
    "Content-Type: application/octet-stream$LF",
    [System.Text.Encoding]::UTF8.GetString($fileBytes),
    "--$boundary",
    "Content-Disposition: form-data; name=`"language`"$LF",
    $language,
    "--$boundary--$LF"
)

$body = $bodyLines -join $LF

try {
    $response = Invoke-WebRequest -Uri $endpoint -Method Post -ContentType "multipart/form-data; boundary=$boundary" -Body $body
    Write-Host "Status Code: $($response.StatusCode)"
    Write-Host "Response: $($response.Content)"
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