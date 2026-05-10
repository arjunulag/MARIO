## How to run

### 1. Backend

Use Python 3.10 and working package versions.

```bash
cd web_mario/backend
py -3.10 -m venv .venv310 (if using a virtual enviroment)
.venv310\Scripts\activate (if using a virtual enviroment)
pip install -r requirements.txt
uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

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
