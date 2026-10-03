import ast
import html
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

st.set_page_config(
    page_title="Reel Finder | Movie recommendations",
    page_icon="R",
    layout="wide",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root {
        --ink: #171b18;
        --muted: #a5aaa2;
        --paper: #f2f0e9;
        --line: rgba(242, 240, 233, .14);
        --acid: #d5f36a;
        --rust: #e47754;
    }
    .stApp {
        background: radial-gradient(ellipse at 78% 4%, rgba(112, 132, 60, .18), transparent 35%),
                    linear-gradient(145deg, #171b18 0%, #202521 58%, #181b19 100%);
        color: var(--paper);
        font-family: 'DM Sans', sans-serif;
    }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: 0 !important; }
    .block-container { max-width: 1100px; padding-top: 3rem; padding-bottom: 4rem; }
    .eyebrow {
        color: var(--acid); font-size: .76rem; font-weight: 700;
        letter-spacing: 0; text-transform: uppercase;
    }
    .hero-title { margin: .3rem 0 0; font-size: 3.75rem; line-height: 1.02; }
    .hero-copy { color: #bdc2b9; font-size: 1.04rem; max-width: 620px; margin-top: .8rem; }
    .section-label {
        border-top: 1px solid var(--line); padding-top: 1.15rem; margin: 2.6rem 0 1rem;
        color: var(--muted); font-size: .76rem; font-weight: 700; letter-spacing: 0;
        text-transform: uppercase;
    }
    .movie-card {
        min-height: 156px; height: 100%; padding: 1.2rem 1.25rem;
        border: 1px solid var(--line); border-radius: 6px;
        background: rgba(242, 240, 233, .045);
    }
    .movie-rank { color: var(--acid); font-size: .76rem; font-weight: 700; letter-spacing: 0; }
    .movie-name { color: var(--paper); font: 600 1.12rem 'Space Grotesk', sans-serif; margin: .6rem 0 .35rem; }
    .movie-meta { color: #a5aaa2; font-size: .84rem; line-height: 1.5; }
    .match { color: var(--rust); font-weight: 700; }
    div[data-testid="stForm"] { border: 1px solid var(--line); border-radius: 6px; background: rgba(0,0,0,.13); }
    div.stButton > button[kind="primary"] {
        background: var(--acid); color: var(--ink); border: 0; font-weight: 700;
        min-height: 2.8rem; border-radius: 4px;
    }
    div.stButton > button[kind="primary"]:hover { background: #e3ff80; color: var(--ink); }
    [data-testid="stSelectbox"] label, [data-testid="stSlider"] label { color: var(--paper); }
    [data-testid="stCaptionContainer"] { color: var(--muted); }
    @media (max-width: 640px) {
        .block-container { padding-top: 2rem; }
        .hero-title { font-size: 2.7rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_names(value, limit=None):
    """Read TMDB's serialized lists and return their name fields."""
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    names = [item["name"]
             for item in parsed if isinstance(item, dict) and item.get("name")]
    return names if limit is None else names[:limit]


def parse_director(value):
    if not isinstance(value, str):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return []
    if not isinstance(parsed, list):
        return []
    return [
        person["name"]
        for person in parsed
        if isinstance(person, dict) and person.get("job") == "Director" and person.get("name")
    ][:1]


@st.cache_resource(show_spinner="Building the movie index...")
def load_recommender():
    movies_path = DATA_DIR / "tmdb_5000_movies.csv"
    credits_path = DATA_DIR / "tmdb_5000_credits.csv"
    if not movies_path.is_file() or not credits_path.is_file():
        raise FileNotFoundError(
            "Place both TMDB CSV files in the project's data/ folder.")

    movies = pd.read_csv(movies_path).merge(
        pd.read_csv(credits_path), on="title")
    required = ["movie_id", "title", "genres",
                "keywords", "overview", "cast", "crew"]
    movies = movies.dropna(subset=required).copy()

    movies["genres_list"] = movies["genres"].map(parse_names)
    movies["keywords_list"] = movies["keywords"].map(parse_names)
    movies["cast_list"] = movies["cast"].map(
        lambda value: parse_names(value, limit=3))
    movies["director_list"] = movies["crew"].map(parse_director)
    movies["overview_words"] = movies["overview"].map(
        lambda value: value.split() if isinstance(value, str) else [])

    tag_columns = ["genres_list", "director_list",
                   "cast_list", "overview_words", "keywords_list"]
    movies["tags"] = movies[tag_columns].apply(
        lambda row: " ".join(
            str(item).replace(" ", "")
            for column in tag_columns
            for item in row[column]
        ).lower(),
        axis=1,
    )

    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = normalize(vectorizer.fit_transform(movies["tags"]))
    return movies.reset_index(drop=True), vectors


st.markdown('<div class="eyebrow">TMDB 5000 · Content-based discovery</div>',
            unsafe_allow_html=True)
st.markdown('<h1 class="hero-title">Find your next<br>favorite film.</h1>',
            unsafe_allow_html=True)
st.markdown(
    '<p class="hero-copy">Pick a movie you love. We’ll find titles with the closest mix of story, cast, genres, and keywords.</p>',
    unsafe_allow_html=True,
)

try:
    movies, vectors = load_recommender()
except Exception as error:
    st.error(f"Could not load the movie recommender: {error}")
    st.stop()


def movie_option_label(row_index):
    movie = movies.iloc[row_index]
    release_date = movie.get("release_date")
    year = release_date[:4] if isinstance(
        release_date, str) and len(release_date) >= 4 else ""
    return f"{movie['title']} ({year})" if year else str(movie["title"])


with st.form("recommendation_form"):
    first_row, second_row = st.columns([4, 1])
    with first_row:
        selected_index = st.selectbox(
            "Choose a movie",
            options=movies.index.tolist(),
            index=None,
            format_func=movie_option_label,
            placeholder="Search the catalog...",
        )
    with second_row:
        result_count = st.slider(
            "Recommendations", min_value=3, max_value=10, value=5)
    submitted = st.form_submit_button(
        "Find similar films", type="primary", use_container_width=True)

if submitted:
    if selected_index is None:
        st.warning("Choose a movie to get recommendations.")
    else:
        selected_movie = movies.iloc[selected_index]
        scores = vectors.getrow(selected_index).dot(
            vectors.T).toarray().ravel()
        scores[selected_index] = -1
        top_indices = scores.argsort()[::-1][:result_count]
        st.markdown(
            '<div class="section-label">Because you chose</div>', unsafe_allow_html=True)
        st.markdown(f"### {selected_movie['title']}")
        st.markdown(
            '<div class="section-label">Your next watchlist</div>', unsafe_allow_html=True)

        for start in range(0, len(top_indices), 2):
            columns = st.columns(2, gap="medium")
            for offset, column in enumerate(columns):
                position = start + offset
                if position >= len(top_indices):
                    continue
                row_index = top_indices[position]
                movie = movies.iloc[row_index]
                genres = " · ".join(
                    movie["genres_list"][:3]) or "Genre not listed"
                directors = ", ".join(
                    movie["director_list"]) or "Director not listed"
                with column:
                    st.markdown(
                        f'''<div class="movie-card">
                            <div class="movie-rank">PICK {position + 1:02d} <span class="match">· {scores[row_index]:.0%} MATCH</span></div>
                            <div class="movie-name">{html.escape(str(movie["title"]))}</div>
                            <div class="movie-meta">{html.escape(genres)}<br>Directed by {html.escape(directors)}</div>
                        </div>''',
                        unsafe_allow_html=True,
                    )
else:
    st.markdown('<div class="section-label">Ready when you are</div>',
                unsafe_allow_html=True)
    st.caption(f"{len(movies):,} movies indexed from the TMDB 5000 dataset.")
