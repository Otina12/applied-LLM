
# Project Setup Guide

This document explains how to set up the environment and run the full multi-agent AutoML pipeline.

## 1. Requirements
You need Python 3.10 or newer. The project runs on Windows, Linux, and macOS.

A dataset must be placed at:

```
data/raw_data.csv
```

A default Titanic dataset exists here:
https://github.com/dsindy/kaggle-titanic/blob/master/data/train.csv  
You may use it or replace it with your own CSV file.

## 2. Environment Setup

You can set up the environment on Windows, macOS, or Linux.  
Only the activation command differs.

### 1. Create a virtual environment

Windows:

```
python -m venv venv
```

macOS or Linux:

```
python3 -m venv venv
```

### 2. Activate the environment

Windows:

```
venv\Scripts\activate
```

macOS or Linux:

```
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

## 3. Environment Variables

Create a `.env` file in the project root directory and place your OpenAI API key there.

```
OPENAI_API_KEY=your_key_here
```

Make sure this file is **never** committed to Git.

## 4. Running the Pipeline

After the environment setup and placing your CSV at:

```
data/raw_data.csv
```

open and run:

```
run_pipeline.ipynb
```

This notebook will start the Data Cleaner agent, the Feature Engineer agent, and the Model Trainer agent.  
It will produce:

- `data/clean_data.csv`
- `data/engineered_data.csv`
- `data/training_report.json`
- `data/engineering_report.json`
- `data/logs.txt`

You can experiment with different datasets and extend the agents as needed.
