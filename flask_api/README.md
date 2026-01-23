# Disease Prediction Flask API

## Setup Instructions

1. **Create a virtual environment:**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Mac/Linux:
     ```bash
     source venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Flask API:**
   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`

## API Endpoints

- `GET /` - API information
- `GET /symptoms` - Get list of all symptoms
- `POST /predict` - Predict disease based on symptoms
- `GET /health` - Health check endpoint

## Example Request

```json
POST http://localhost:5000/predict
{
  "symptoms": {
    "fever": true,
    "cough": true,
    "fatigue": true,
    "difficulty_breathing": false,
    "headache": true
  }
}
```

## Example Response

```json
{
  "disease": "Flu",
  "confidence": 85.5,
  "symptoms_checked": 3,
  "total_symptoms": 20
}
```
