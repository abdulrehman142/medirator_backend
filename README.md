# Medirator Backend

FastAPI backend for Medirator.

## Requirements

- Python 3.11+ recommended
- MongoDB for local development or a MongoDB Atlas connection for deployment
- Redis for rate limiting/session support

## Setup

Create the virtual environment and install dependencies:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Copy the example environment file and adjust it for your setup:

```bash
cp .env.example .env
```

## MongoDB Setup

### Local MongoDB

On macOS with Homebrew:

```bash
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community
```

Then update `.env` with:

```env
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=medirator_db
```

### Deployed MongoDB Atlas

To use a deployed MongoDB Atlas instance, update `MONGO_URI` in `.env`:

```env
MONGO_URI=mongodb+srv://USERNAME:PASSWORD@cluster0.example.mongodb.net/medirator_db?retryWrites=true&w=majority
MONGO_DB_NAME=medirator_db
```

Replace `USERNAME`, `PASSWORD`, and the cluster address with your actual MongoDB Atlas credentials. You can find this connection string in your MongoDB Atlas dashboard under "Connect" → "Connect your application".

## Running The Backend

Ensure MongoDB and Redis are running first. From the repository root:

```bash
# Activate the virtual environment
source .venv/bin/activate

# Navigate to backend directory
cd backend

# Start the server using the venv Python
../.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

**Important:** Use the full path to the venv Python (`../.venv/bin/python`) to ensure all dependencies are properly resolved.

Health checks:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/health/db
```

## Notes

- `/health` should return `{"status":"ok"}` once the app is running.
- `/health/db` returns `503` until MongoDB is reachable.
- The app now starts even if MongoDB is unavailable, but database-backed features will still require a working MongoDB connection.

## Render Deployment

This repository includes a Render blueprint at the repository root.

Render service settings:

```yaml
buildCommand: pip install -r requirements.txt
startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
rootDir: backend
```

Recommended Render environment variables:

```env
APP_NAME=Medirator API
ENV=production
DEBUG=false
API_V1_PREFIX=/api/v1
MONGO_URI=<your MongoDB Atlas URI>
MONGO_DB_NAME=medirator_db
REDIS_URL=<your Redis URL>
JWT_SECRET_KEY=<32+ char secret>
ALLOWED_ORIGINS=https://your-frontend-domain.com
GEMINI_API_KEY=<optional>
HUGGINGFACE_HUB_TOKEN=<optional>
XRAY_MODEL_URL=<optional model URL>
SYMPTOM_MODEL_URL=<optional model URL>
SYMPTOMS_LIST_URL=<optional model URL>
LABEL_ENCODER_URL=<optional model URL>
```

If you want Render to manage the service directly from the repo, use the `render.yaml` file in the project root. The backend still needs a reachable MongoDB instance, and Redis is recommended for rate limiting and auth/session flows.
