# Movie Recommendation System

A content-based movie recommender built with Streamlit and the TMDB 5000 dataset.

## Run locally

From the project root, create and activate a virtual environment, then install dependencies:

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The app expects both `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` in the `data/` directory.

## Deploy to Streamlit Community Cloud

1. Push this project to a GitHub repository, including `app.py`, `requirements.txt`, and both files in `data/`.
2. In Streamlit Community Cloud, create an app from that repository.
3. Set the main file path to `app.py` and deploy.

The movie index is built on the first app load and cached for subsequent interactions.