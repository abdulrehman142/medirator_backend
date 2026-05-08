brew services start mongodb-community
cd backend
/opt/homebrew/opt/python@3.11/bin/python3.11 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --app-dir .
# Note
# Use the root venv located at `./.venv` from the repository root. The `backend/.venv` directory is a separate environment and may not have all required packages.
