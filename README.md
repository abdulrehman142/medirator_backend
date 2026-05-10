# Medirator Backend Repository

This repository contains the backend FastAPI service in the `backend/` directory.

ML model loaders and training/demo scripts are kept in the `ML/` folder so they stay out of the backend deployment.

## Local Run

From the repository root:

```bash
cd backend
python -m uvicorn app.main:app --reload
```

If you are using the existing virtual environment, this also works:

```bash
cd backend
../.venv/bin/python -m uvicorn app.main:app --reload
```

## Render Deploy

Use `backend/` as the service root and set the start command to:

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Render should install dependencies from `backend/requirements.txt`, which keeps the deployment backend-only and avoids relying on a globally installed `uvicorn`.

## Environment Variables

Do not commit `.env` files. Set variables in your shell locally or in the Render dashboard. The repo keeps only `.env.example` as a template.
