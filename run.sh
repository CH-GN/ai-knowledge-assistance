#!/bin/bash

source venv/bin/activate
uvicorn main:app --host 127.0.0.1 --port 5000 --reload