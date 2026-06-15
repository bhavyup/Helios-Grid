if (-not (Test-Path "backend\artifacts")) {
  New-Item -ItemType Directory -Force backend\artifacts | Out-Null
}

$authBodyPath = "backend\artifacts\auth_body.json"
$resetBodyPath = "backend\artifacts\reset_body.json"

'{"email":"dev@helios.local","password":"dev-pass-123"}' | Set-Content -Path $authBodyPath
'{"seed":123,"num_households":64,"max_episode_steps":8}' | Set-Content -Path $resetBodyPath

$register = curl.exe -s -X POST http://localhost:8000/auth/register -H "Content-Type: application/json" -d "@$authBodyPath"
$token = ($register | ConvertFrom-Json).access_token
if (-not $token) {
  $login = curl.exe -s -X POST http://localhost:8000/auth/login -H "Content-Type: application/json" -d "@$authBodyPath"
  $token = ($login | ConvertFrom-Json).access_token
}
if (-not $token) {
  throw "Failed to obtain access token"
}

$reset = curl.exe -s -X POST http://localhost:8000/simulation/reset -H "Content-Type: application/json" -H "Authorization: Bearer $token" -d "@$resetBodyPath"
$reset | Set-Content -Path backend\artifacts\reset_64.json

$json = $reset | ConvertFrom-Json
$houses = $json.topology.nodes | Where-Object { $_.type -eq "household" }
$unique = $houses | ForEach-Object { "{0:N3},{1:N3}" -f $_.x, $_.y } | Sort-Object -Unique
Write-Host "houses: $($houses.Count) unique: $($unique.Count)"
