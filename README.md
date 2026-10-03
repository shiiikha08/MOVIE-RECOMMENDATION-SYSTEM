# Reel Finder

A content-based movie recommender built with Python, Streamlit, and the TMDB 5000 movie dataset. Choose a film and Reel Finder ranks other films by how closely their genres, story, cast, director, and keywords match.

## Features

- Search a catalog of more than 4,800 movies by title and release year.
- Choose how many recommendations to see, from 3 to 10.
- View ranked recommendations with a similarity score, genres, and director.
- Run the recommender locally without an API key.

## How recommendations work

The app combines each movie's genres, director, up to three cast members, overview, and keywords into a text profile. It converts those profiles into word-count vectors with scikit-learn, normalizes the vectors, and compares the selected movie with the rest of the catalog using cosine similarity. The selected movie is excluded from its own results.

The displayed match percentage is a relative text-similarity score. It is not a user rating, review score, or guarantee that you will enjoy a recommendation.

The movie index is built on the first app run and cached for later interactions in the same session.

## Run locally

Use Python 3.10 or newer. From the project directory, create a virtual environment and install the dependencies.

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Streamlit prints a local URL when the app starts, usually `http://localhost:8501`.

## Dataset

The app reads these files from `data/`:

- `tmdb_5000_movies.csv`
- `tmdb_5000_credits.csv`

Both files are required. The app merges them on movie title and builds its index at startup. Keep the dataset's original attribution and terms with any redistribution.

## Project structure

```text
.
├── app.py                         # Streamlit interface and recommendation pipeline
├── data/
│   ├── tmdb_5000_credits.csv      # Cast and crew metadata
│   └── tmdb_5000_movies.csv       # Movie metadata
├── notebooks/
│   └── notebook.ipynb             # Exploratory model development
├── requirements.txt               # Python dependencies
└── README.md
```

## Dependencies

Dependencies are listed in [`requirements.txt`](requirements.txt): Streamlit, pandas, and scikit-learn. A local `.venv/` is excluded from version control.