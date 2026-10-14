# AI Learning Assistant 2.0

## Run it
1. Install Python 3.10 or newer.
2. Open a terminal in this folder.
3. Install dependencies:
   `pip install -r requirements.txt`
4. Start the app:
   `streamlit run app.py`

The app opens in your browser. It works without a dataset using generated demo data. Optional CSV upload supports student activity data.

Expected CSV columns (recommended):
- student_id (optional)
- study_hours
- completion_rate (0–1)
- quiz_attempts
- interactions
- help_count
- attendance_rate (0–1)
- average_score (0–1 or 0–100)
- performance_level (optional)
- subject (optional)

The AI Tutor retrieves relevant text from uploaded TXT/PDF notes using TF-IDF. It is not a generative LLM and does not send files to an external service.
