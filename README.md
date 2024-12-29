# Enterprise ML Server

A production-ready ML server for serving XGBoost models using FastAPI.

## Features

- FastAPI backend for high performance
- XGBoost model serving
- Configuration management
- API key authentication
- Use FastAPI for serving models
- Use Pydantic to validate incoming parameters. 
- Health checks
- Comprehensive testing
- Logging
- Error handling
- Type safety with Pydantic
- Environment variable configuration

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

4. Place your XGBoost model in the `models` directory.

## Running the Server

```bash
python src/main.py
```

## API Endpoints

- `GET /api/v1/health` - Health check
- `POST /api/v1/predict` - Make predictions

## Testing

```bash
python -m pytest tests/
```

## Configuration

See `.env.example` for available configuration options.