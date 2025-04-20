Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m uvicorn app.main:app --host 0.0.0.0 --port 8000" -RedirectStandardOutput "api-server.log"

Write-Host "API sunucusu başlatıldı, çıktılar api-server.log dosyasına yönlendirildi."
Write-Host "Sunucuyu durdurmak için Stop-Process komutu ile işlemi sonlandırın:"
Write-Host "Get-Process -Name python | Where-Object {`$_.CommandLine -like '*uvicorn*'} | Stop-Process" 