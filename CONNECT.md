# How to Connect the Pipeline -> Flask API -> React Dashboard

## Architecture

```
pipeline.py  --writes--  outputs/
                              -- backtest/  (equity curves, metrics CSVs)
                              -- artifacts/ (saved models)
                              -- signals_YYYY-MM-DD.csv
                              -- *_model_comparison.csv
                                        
                                    app.py  (Flask API reads all of the above)
                                        
                                   :5000/api/*
                                        
                              dashboard_app.jsx  (React fetches from API)
```

---

## Step 1 - Install Flask dependencies

```bash
pip install flask flask-cors scipy xgboost
```

---

## Step 2 - Run the training pipeline (if not done yet)

```bash
python pipeline.py --mode train --tickers RELIANCE.NS TCS.NS INFY.NS
```

This writes all output files that the API reads.

---

## Step 3 - Start the Flask API

```bash
python app.py
```

You should see:
```
AlgoTrader API  ->  http://localhost:5000
```

Test it works:
```
http://localhost:5000/api/tickers
http://localhost:5000/api/price/RELIANCE.NS
http://localhost:5000/api/metrics/RELIANCE.NS
http://localhost:5000/api/backtest/RELIANCE.NS
http://localhost:5000/api/signals
```

---

## Option A - React Dev Server (development)

### Install Node.js if you don't have it
Download from https://nodejs.org (LTS version)

### Create a React app

```bash
npx create-react-app dashboard
cd dashboard
npm install recharts
```

### Replace src/App.js with the dashboard

Copy `dashboard_app.jsx` content into `dashboard/src/App.js`

### Start the dev server

```bash
npm start
```

Opens at http://localhost:3000
It fetches data from Flask at http://localhost:5000

---

## Option B - Serve React from Flask (production / single server)

```bash
cd dashboard
npm run build
cd ..
```

Flask will now serve the built React app at http://localhost:5000
(The `serve_react` route in app.py handles this automatically)

---

## Step 4 - Generate today's signals

Either from the command line:
```bash
python pipeline.py --mode predict --tickers RELIANCE.NS TCS.NS INFY.NS
```

Or click **" Generate Today's Signals"** in the Pipeline tab of the dashboard.

---

## Step 5 - Run everything at once (convenience script)

Create `start.bat` (Windows):
```bat
@echo off
start "Flask API" python app.py
timeout /t 3
start "Dashboard" cmd /c "cd dashboard && npm start"
```

Create `start.sh` (Mac/Linux):
```bash
#!/bin/bash
python app.py &
cd dashboard && npm start
```

---

## Data Flow Summary

| Dashboard Tab | API Endpoint              | Source File                          |
|---------------|---------------------------|--------------------------------------|
| Overview      | /api/price/<ticker>       | data/processed/<ticker>_features.parquet |
| Overview      | /api/metrics/<ticker>     | outputs/<ticker>_model_comparison.csv |
| Backtest      | /api/backtest/<ticker>    | outputs/backtest/<ticker>_equity.csv |
| Features      | /api/features/<ticker>    | outputs/artifacts/<ticker>_rf.pkl    |
| Features      | /api/eda/<ticker>         | computed live from parquet           |
| Signals       | /api/signals              | outputs/signals_YYYY-MM-DD.csv       |
| Pipeline      | /api/pipeline/run  (POST) | triggers pipeline.py subprocess      |
| Pipeline      | /api/pipeline/status      | in-memory log stream                 |

---

## Troubleshooting

**"API Connected" shows red / fetch errors**
-> Make sure `python app.py` is running in a separate terminal

**No price data / metrics**
-> Run the training pipeline first: `python pipeline.py --mode train ...`

**No signals**
-> Run predict mode: `python pipeline.py --mode predict ...`
   Or click "Generate Signals" in the Pipeline tab

**CORS error in browser console**
-> flask-cors is installed and enabled in app.py - restart Flask

**Port conflict**
-> Change `port=5000` in `app.py` and update `const API = "http://localhost:XXXX/api"` in the JSX
