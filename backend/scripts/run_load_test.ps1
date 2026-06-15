param(
    [string]$HostUrl = "http://127.0.0.1:8000",
    [int]$Users = 100,
    [int]$SpawnRate = 10,
    [string]$RunTime = "5m",
    [string]$Username = "loadtest@example.com",
    [string]$Password = "test-pass-123",
    [int]$SimulationBurstSize = 1,
    [int]$SimulationHouseholds = 32,
    [int]$SimulationRunSteps = 5,
    [int]$TrainingEpisodes = 2,
    [int]$TrainingStepsPerEpisode = 4,
    [int]$TrainingEvalEpisodes = 1,
    [switch]$TrainingWaitForResult
)

$env:LOCUST_HOST = $HostUrl
$env:LOCUST_USERNAME = $Username
$env:LOCUST_PASSWORD = $Password
$env:SIMULATION_BURST_SIZE = $SimulationBurstSize
$env:SIMULATION_HOUSEHOLDS = $SimulationHouseholds
$env:SIMULATION_RUN_STEPS = $SimulationRunSteps
$env:TRAINING_EPISODES = $TrainingEpisodes
$env:TRAINING_STEPS_PER_EPISODE = $TrainingStepsPerEpisode
$env:TRAINING_EVAL_EPISODES = $TrainingEvalEpisodes
if ($TrainingWaitForResult) {
    $env:TRAINING_WAIT_FOR_RESULT = "true"
}

$locustExe = Join-Path $PSScriptRoot "..\.venv\Scripts\locust.exe"
if (-not (Test-Path $locustExe)) {
    $locustExe = "locust"
}

& $locustExe -f (Join-Path $PSScriptRoot "..\load_tests\locustfile.py") --headless -u $Users -r $SpawnRate -t $RunTime
