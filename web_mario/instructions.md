## How to run

### 1. Backend

Use Python 3.10 and working package versions.

```bash
cd web_mario/backend
py -3.10 -m venv .venv310 (if using a virtual enviroment)
.venv310\Scripts\activate (if using a virtual enviroment)
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

Use `--reload` only while editing backend code. For playing, leave it off so the
backend does less background work.

### 2. Frontend

Open a second terminal:

```bash
cd web_mario/frontend
python -m http.server 3000
```

Then open:

```text
http://127.0.0.1:3000
```

For slower laptops, use the website's Mode dropdown:

- Stable 36: default, lower CPU/GPU load.
- Laptop 24: lowest load.
- Competition 60: full-speed runs for leaderboard-style comparisons.
