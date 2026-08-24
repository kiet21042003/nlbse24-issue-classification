$ErrorActionPreference = "Stop"

ruff check src tests
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

pytest --cov=nlbse24 --cov-report=term-missing --cov-fail-under=80
exit $LASTEXITCODE
