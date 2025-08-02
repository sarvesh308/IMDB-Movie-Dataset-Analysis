import streamlit as st
import sqlite3
import pandas as pd
import hashlib
import plotly.express as px
from fpdf import FPDF
import os
import tempfile
import textwrap

# --- Constants ---
DB_PATH = 'imdb_data.db'
CSV_PATH = 'imdb.csv'

# --- Page Setup ---
st.set_page_config("🎬 IMDB Visual Analyzer", layout="wide", initial_sidebar_state="expanded")

# --- Styling (static light theme) ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');

    html, body, [class*="css"] {
        background: #f5f8fc !important;
        color: #1e2022 !important;
        font-family: 'Montserrat', sans-serif;
        transition: background .3s ease, color .3s ease;
    }
    .stSidebar .sidebar-content {
        background: linear-gradient(135deg, #1f4e8c, #3f7acb) !important;
        padding: 1rem;
        border-radius: 14px;
    }
    h1, h2 {
        position: relative;
        display: inline-block;
        padding-bottom: 6px;
        margin-bottom: 8px;
    }
    h1:after, h2:after {
        content: "";
        position: absolute;
        left: 0;
        bottom: 0;
        width: 60px;
        height: 4px;
        border-radius: 4px;
        background: linear-gradient(90deg,#0077b6,#00aaff);
    }
    .card {
        background: #ffffff;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 28px 55px -10px rgba(0,0,0,0.12);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .stButton>button {
        background: #0077b6 !important;
        color: white !important;
        border-radius: 12px;
        padding: 0.65rem 1.3rem;
        font-weight: 600;
        border: none;
        box-shadow: 0 12px 30px rgba(0,119,182,0.25);
        transition: all .2s ease-in-out;
    }
    .stButton>button:hover {
        transform: scale(1.03);
        box-shadow: 0 16px 40px rgba(0,119,182,0.4);
    }
    .stSelectbox>div, .stTextInput>div>div>input {
        border-radius: 10px;
        padding: 6px;
    }
    .dataframe tbody tr:hover {
        background-color: rgba(0, 119, 182, 0.08) !important;
    }
    .small-label {
        font-size:12px; color: #555;
    }
    .divider {
        height:2px;
        background: linear-gradient(90deg,#0077b6,#00aaff);
        border-radius:3px;
        margin:12px 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- Database helpers ---
def get_db_conn():
    return sqlite3.connect(DB_PATH)

def make_hashes(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def create_user_table():
    conn = get_db_conn()
    conn.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT UNIQUE, password TEXT)')
    conn.commit()
    conn.close()

def add_user(username: str, password: str) -> bool:
    conn = get_db_conn()
    try:
        conn.execute('INSERT INTO userstable(username, password) VALUES (?, ?)', (username, make_hashes(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def login_user(username: str, password: str) -> bool:
    conn = get_db_conn()
    cur = conn.execute('SELECT * FROM userstable WHERE username = ? AND password = ?', (username, make_hashes(password)))
    res = cur.fetchone() is not None
    conn.close()
    return res

def ensure_movies_table():
    conn = get_db_conn()
    cursor = conn.cursor()
    # Check if table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='movies'")
    if not cursor.fetchone():
        # Load from CSV
        if not os.path.exists(CSV_PATH):
            st.error(f"CSV data file '{CSV_PATH}' not found. Cannot populate movies table.")
            conn.close()
            return
        df_csv = pd.read_csv(CSV_PATH)
        df_csv.columns = [c.lower().strip() for c in df_csv.columns]
        # Write to sqlite
        df_csv.to_sql('movies', conn, if_exists='replace', index=False)
    else:
        # if exists but empty, reload
        count = cursor.execute("SELECT COUNT(1) FROM movies").fetchone()[0]
        if count == 0:
            if os.path.exists(CSV_PATH):
                df_csv = pd.read_csv(CSV_PATH)
                df_csv.columns = [c.lower().strip() for c in df_csv.columns]
                df_csv.to_sql('movies', conn, if_exists='replace', index=False)
    conn.commit()
    conn.close()

@st.cache_data
def load_movies_df():
    conn = get_db_conn()
    df = pd.read_sql_query("SELECT * FROM movies", conn)
    conn.close()
    # ensure consistent column names
    df.columns = df.columns.str.lower().str.strip()
    return df

def fetch_query(selected_q, genre=None, year=None, year_range=None):
    conn = get_db_conn()
    result = pd.DataFrame()
    chart = None
    try:
        if selected_q == "Top 10 Rated Movies":
            result = pd.read_sql_query("SELECT * FROM movies ORDER BY rating DESC LIMIT 10", conn)
            chart = px.bar(result, x="rating", y="title", orientation="h",
                           title="Top Movies by Rating", template="plotly_white", color="rating")
        elif selected_q == "Top 10 Revenue Movies":
            result = pd.read_sql_query("SELECT * FROM movies ORDER BY revenue DESC LIMIT 10", conn)
            chart = px.bar(result, x="revenue", y="title", orientation="h",
                           title="Top Revenue Movies", template="plotly_white", color="revenue")
        elif selected_q == "Top Directors":
            result = pd.read_sql_query("""
                SELECT director, COUNT(*) AS count
                FROM movies
                WHERE director IS NOT NULL
                GROUP BY director
                ORDER BY count DESC
                LIMIT 10
            """, conn)
            result.columns = ["Director", "Count"]
            chart = px.bar(result, x="Count", y="Director", orientation="h",
                           title="Top Directors", template="plotly_white", color="Count")
        elif selected_q == "Movies by Genre" and genre:
            result = pd.read_sql_query("SELECT * FROM movies WHERE genre LIKE ? COLLATE NOCASE", conn, params=(f"%{genre}%",))
        elif selected_q == "Movies after 2010":
            result = pd.read_sql_query("SELECT * FROM movies WHERE year > 2010", conn)
        elif selected_q == "Short Runtime Movies":
            result = pd.read_sql_query("SELECT * FROM movies WHERE runtime < 100 ORDER BY runtime ASC", conn)
        elif selected_q == "Long Runtime Movies":
            result = pd.read_sql_query("SELECT * FROM movies WHERE runtime > 150 ORDER BY runtime DESC", conn)
        elif selected_q == "Movies with Most Votes":
            result = pd.read_sql_query("SELECT * FROM movies ORDER BY votes DESC LIMIT 10", conn)
            chart = px.bar(result, x="votes", y="title", orientation="h",
                           title="Movies with Most Votes", template="plotly_white", color="votes")
        elif selected_q == "Low Rated Movies":
            result = pd.read_sql_query("SELECT * FROM movies ORDER BY rating ASC LIMIT 10", conn)
            chart = px.bar(result, x="rating", y="title", orientation="h",
                           title="Low Rated Movies", template="plotly_white", color="rating")
        elif selected_q == "Top Movies per Year" and year:
            result = pd.read_sql_query("SELECT * FROM movies WHERE year = ? ORDER BY rating DESC LIMIT 10", conn, params=(year,))
        elif selected_q == "Top 10 Genres":
            result = pd.read_sql_query("""
                SELECT genre, COUNT(*) as count
                FROM movies
                WHERE genre IS NOT NULL
                GROUP BY genre
                ORDER BY count DESC
                LIMIT 10
            """, conn)
            result.columns = ["Genre", "Count"]
            chart = px.bar(result, x="Count", y="Genre", orientation="h",
                           title="Top Genres", template="plotly_white", color="Count")
        elif selected_q == "Revenue by Genre":
            result = pd.read_sql_query("""
                SELECT genre, SUM(revenue) as revenue
                FROM movies
                WHERE genre IS NOT NULL
                GROUP BY genre
                ORDER BY revenue DESC
                LIMIT 10
            """, conn)
            result = result.rename(columns={"genre": "Genre", "revenue": "Revenue"})
            chart = px.bar(result, x="Revenue", y="Genre", orientation="h",
                           title="Revenue by Genre", template="plotly_white", color="Revenue")
        elif selected_q == "Average Rating by Genre":
            result = pd.read_sql_query("""
                SELECT genre, AVG(rating) as avg_rating
                FROM movies
                WHERE genre IS NOT NULL
                GROUP BY genre
                ORDER BY avg_rating DESC
                LIMIT 10
            """, conn)
            result = result.rename(columns={"genre": "Genre", "avg_rating": "Avg_Rating"})
            chart = px.bar(result, x="Avg_Rating", y="Genre", orientation="h",
                           title="Average Rating by Genre", template="plotly_white", color="Avg_Rating")
        elif selected_q == "Director with Most Votes":
            result = pd.read_sql_query("""
                SELECT director, SUM(votes) as votes
                FROM movies
                WHERE director IS NOT NULL
                GROUP BY director
                ORDER BY votes DESC
                LIMIT 10
            """, conn)
            result = result.rename(columns={"director": "Director", "votes": "Votes"})
            chart = px.bar(result, x="Votes", y="Director", orientation="h",
                           title="Director by Votes", template="plotly_white", color="Votes")
        elif selected_q == "Top Movies by Runtime":
            result = pd.read_sql_query("SELECT * FROM movies ORDER BY runtime DESC LIMIT 10", conn)
        elif selected_q == "Movies Between Years" and year_range:
            result = pd.read_sql_query("SELECT * FROM movies WHERE year BETWEEN ? AND ?", conn,
                                       params=(year_range[0], year_range[1]))
        elif selected_q == "Top Movies by Votes-to-Rating Ratio":
            # avoid division by zero
            result = pd.read_sql_query("""
                SELECT *,
                    CASE WHEN rating = 0 THEN 0.0 ELSE CAST(votes AS REAL)/rating END AS ratio
                FROM movies
                ORDER BY ratio DESC
                LIMIT 10
            """, conn)
            chart = px.bar(result, x="ratio", y="title", orientation="h",
                           title="Top Movies by Votes-to-Rating Ratio", template="plotly_white", color="ratio")
        elif selected_q == "Top 5 Movies per Genre" and genre:
            result = pd.read_sql_query("SELECT * FROM movies WHERE genre LIKE ? COLLATE NOCASE ORDER BY rating DESC LIMIT 5", conn,
                                       params=(f"%{genre}%",))
        elif selected_q == "Oldest Movies":
            result = pd.read_sql_query("SELECT * FROM movies ORDER BY year ASC LIMIT 10", conn)
        elif selected_q == "Most Recent Movies":
            result = pd.read_sql_query("SELECT * FROM movies ORDER BY year DESC LIMIT 10", conn)
        else:
            # fallback: entire table if nothing matched
            result = pd.read_sql_query("SELECT * FROM movies LIMIT 100", conn)
    except Exception as e:
        st.error(f"Query error: {e}")
    finally:
        conn.close()
    return result, chart

# --- Fallback chart generator ---
def make_fallback_chart(df):
    numeric = df.select_dtypes(include='number').columns.tolist()
    categorical = df.select_dtypes(include='object').columns.tolist()
    if numeric and categorical:
        x = numeric[0]
        y = categorical[0]
        subset = df.sort_values(by=x, ascending=False).head(10)
        return px.bar(subset, x=x, y=y, orientation="h", title=f"{y} by {x}", template="plotly_white", color=x)
    elif numeric:
        x = numeric[0]
        subset = df.sort_values(by=x, ascending=False).head(10)
        return px.bar(subset, x=x, y=subset.index.astype(str), orientation="h", title=f"{x} distribution", template="plotly_white", color=x)
    else:
        return None

# --- PDF Export ---
def export_pdf_table_chart(title, df_table, fig, filename="imdb_report.pdf"):
    chart_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    try:
        fig.write_image(chart_tmp.name, engine="kaleido")
    except Exception as e:
        chart_tmp.close()
        raise e
    chart_tmp.close()

    pdf = FPDF(orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=12)
    pdf.add_page()
    page_width = pdf.w - 2 * pdf.l_margin

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, title, ln=True, align="C")
    pdf.ln(5)

    display_df = df_table.copy()
    max_cols = 6
    if display_df.shape[1] > max_cols:
        display_df = display_df.iloc[:, :max_cols]
    num_cols = display_df.shape[1]
    col_width = page_width / num_cols if num_cols > 0 else page_width

    pdf.set_font("Arial", "B", 11)
    pdf.set_fill_color(230, 230, 230)
    for col in display_df.columns:
        pdf.cell(col_width, 8, str(col)[:15], border=1, align="C", fill=True)
    pdf.ln()

    pdf.set_font("Arial", size=9)
    for i in range(min(len(display_df), 12)):
        row = display_df.iloc[i]
        for item in row:
            pdf.cell(col_width, 7, textwrap.shorten(str(item), width=15, placeholder="…"), border=1)
        pdf.ln()

    pdf.ln(4)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, "Visualization", ln=True)
    pdf.ln(2)
    pdf.image(chart_tmp.name, x=pdf.l_margin, w=page_width)
    pdf.ln(5)

    pdf.set_font("Arial", "I", 8)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 10, "Generated by IMDB Visual Analyzer • 2025", align="C")

    output_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    pdf.output(output_tmp.name)
    os.remove(chart_tmp.name)
    return output_tmp.name

# --- Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# --- Setup DB ---
create_user_table()
ensure_movies_table()

# --- Authentication Sidebar ---
with st.sidebar:
    st.subheader("🔐 Account")
    if not st.session_state.logged_in:
        auth_tab = st.radio("Action", ["Login", "Sign Up"])
        if auth_tab == "Sign Up":
            with st.form("signup_form", clear_on_submit=False):
                new_user = st.text_input("Username", key="signup_user")
                new_pass = st.text_input("Password", type="password", key="signup_pass")
                submitted = st.form_submit_button("Create Account")
                if submitted:
                    if new_user and new_pass:
                        success = add_user(new_user, new_pass)
                        if success:
                            st.success("Account created. Please login.")
                        else:
                            st.error("Username already exists.")
                    else:
                        st.warning("Both fields required.")
        else:  # Login
            with st.form("login_form", clear_on_submit=False):
                user = st.text_input("Username", key="login_user")
                passwd = st.text_input("Password", type="password", key="login_pass")
                submitted = st.form_submit_button("Login")
                if submitted:
                    if user and passwd:
                        if login_user(user, passwd):
                            st.session_state.logged_in = True
                            st.session_state.username = user
                            st.success(f"Welcome back, {user}!")
                        else:
                            st.error("Incorrect username or password.")
                    else:
                        st.warning("Both fields required.")
    else:
        st.markdown(f"**Logged in as:** `{st.session_state.username}`")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.username = ""

# --- Main App ---
if st.session_state.logged_in:
    st.markdown(f"<h2>🎉 Welcome, <code>{st.session_state.username}</code></h2>", unsafe_allow_html=True)
    st.markdown("<div class='small-label'>IMDB Visual Analyzer</div>", unsafe_allow_html=True)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # Fetch movie data minimally for directors list etc.
    df_full = load_movies_df()
    if df_full.empty:
        st.warning("No movie data available.")
    else:
        st.markdown("### 🔍 Explore IMDB Movie Data")
        with st.container():
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            questions = [
                "Top 10 Rated Movies", "Top 10 Revenue Movies", "Top Directors",
                "Movies by Genre", "Movies after 2010", "Short Runtime Movies",
                "Long Runtime Movies", "Movies with Most Votes", "Low Rated Movies",
                "Top Movies per Year", "Top 10 Genres", "Revenue by Genre",
                "Average Rating by Genre", "Director with Most Votes", "Top Movies by Runtime",
                "Movies Between Years", "Top Movies by Votes-to-Rating Ratio",
                "Top 5 Movies per Genre", "Oldest Movies", "Most Recent Movies"
            ]
            selected_q = st.selectbox("📌 Choose a Query", questions)
            genre = st.text_input("🎭 Enter Genre") if "Genre" in selected_q else None
            year = st.selectbox("📅 Select Year", sorted(df_full['year'].dropna().unique(), reverse=True)) if "Year" in selected_q and "Between" not in selected_q else None
            year_range = st.slider("📅 Select Year Range", int(df_full['year'].min()), int(df_full['year'].max()), (2000, 2020)) if "Between" in selected_q else None
            view = st.radio("🧭 View As", ["Table", "Plotly Chart"], horizontal=True)
            st.markdown("</div>", unsafe_allow_html=True)

        result, chart = fetch_query(selected_q, genre=genre, year=year, year_range=year_range)

        try:
            if not result.empty:
                if view == "Table":
                    st.dataframe(result, use_container_width=True)
                else:
                    if chart is not None:
                        st.plotly_chart(chart, use_container_width=True)
                    else:
                        fallback = make_fallback_chart(result)
                        if fallback:
                            st.plotly_chart(fallback, use_container_width=True)
                            chart = fallback
                        else:
                            st.warning("No suitable data for visualization.")

                st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
                if st.button("📄 Export to PDF Report"):
                    title = f"IMDB Report - {selected_q}"
                    chosen_chart = chart if chart is not None else (make_fallback_chart(result) or px.bar(result, template="plotly_white"))
                    try:
                        pdf_path = export_pdf_table_chart(title, result, chosen_chart, filename=f"{selected_q.replace(' ', '_')}.pdf")
                        with open(pdf_path, "rb") as f:
                            st.download_button("⬇️ Download PDF", f, file_name=os.path.basename(pdf_path), mime="application/pdf")
                    except Exception as e:
                        st.error(f"Failed to export PDF: {e}")
            else:
                st.warning("⚠️ No data matched your criteria.")
        except Exception as e:
            st.error(f"🚫 Error rendering results: {e}")

        # --- Future Analysis Section ---
        st.markdown("## 🔮 Future Analysis for Directors")
        with st.expander("Director Genre Opportunity & Trends"):
            all_directors = sorted(df_full['director'].dropna().unique().tolist())
            selected_director = st.selectbox("Select Director for Analysis", ["-- choose --"] + all_directors, key="future_director")
            if selected_director and selected_director != "-- choose --":
                director_df = df_full[df_full['director'] == selected_director]
                st.subheader(f"🎬 Genre Profile for {selected_director}")
                genre_count = director_df['genre'].value_counts().reset_index()
                genre_count.columns = ["Genre", "Count"]
                avg_rating_by_genre = director_df.groupby('genre')['rating'].mean().reset_index()
                avg_rating_by_genre.columns = ['Genre', 'Avg_Rating']
                avg_revenue_by_genre = director_df.groupby('genre')['revenue'].mean().reset_index()
                avg_revenue_by_genre.columns = ['Genre', 'Avg_Revenue']
                profile = genre_count.merge(avg_rating_by_genre, on='Genre', how='left').merge(avg_revenue_by_genre, on='Genre', how='left')
                st.dataframe(profile)

                st.subheader("📈 Overall Genre Benchmarks")
                overall_genre_rating = df_full.groupby('genre')['rating'].mean().reset_index()
                overall_genre_rating.columns = ['Genre', 'Overall_Avg_Rating']
                overall_genre_count = df_full['genre'].value_counts().reset_index()
                overall_genre_count.columns = ['Genre', 'Global_Count']
                benchmarks = overall_genre_rating.merge(overall_genre_count, on='Genre')
                st.dataframe(benchmarks.head(10))

                st.subheader("💡 Opportunity Suggestions")
                done_genres = set(director_df['genre'].dropna().unique())
                high_rating_genres = overall_genre_rating[~overall_genre_rating['Genre'].isin(done_genres)].sort_values(by='Overall_Avg_Rating', ascending=False).head(5)
                if not high_rating_genres.empty:
                    st.markdown("Genres you haven't explored much but that have high average ratings overall:")
                    st.table(high_rating_genres.rename(columns={'Genre': 'Genre', 'Overall_Avg_Rating': 'Avg Rating'}))
                else:
                    st.info("No immediate new high-rated genre opportunities found.")

                st.subheader("📊 Genre Rating Trends (Recent vs Prior)")
                current_year = int(df_full['year'].max())
                recent_cutoff = current_year - 5
                prior_cutoff = recent_cutoff - 5

                recent = df_full[df_full['year'] > recent_cutoff]
                prior = df_full[(df_full['year'] <= recent_cutoff) & (df_full['year'] > prior_cutoff)]

                recent_avg = recent.groupby('genre')['rating'].mean().reset_index().rename(columns={'genre': 'Genre', 'rating': 'Recent_Avg'})
                prior_avg = prior.groupby('genre')['rating'].mean().reset_index().rename(columns={'genre': 'Genre', 'rating': 'Prior_Avg'})

                trend = recent_avg.merge(prior_avg, on='Genre', how='inner')
                trend['Delta'] = trend['Recent_Avg'] - trend['Prior_Avg']
                trend = trend.sort_values(by='Delta', ascending=False)
                st.dataframe(trend.head(10))

                st.subheader("🚀 Recommended Next Genres")
                overall_rating = overall_genre_rating.copy()
                trend_df = trend[['Genre', 'Delta']].copy()
                director_genre_counts = director_df['genre'].value_counts().to_dict()
                reco = overall_rating.merge(trend_df, on='Genre', how='left').fillna(0)
                min_r, max_r = reco['Overall_Avg_Rating'].min(), reco['Overall_Avg_Rating'].max()
                reco['Q'] = (reco['Overall_Avg_Rating'] - min_r) / (max_r - min_r + 1e-9)
                reco['T'] = reco['Delta'] / reco['Delta'].max() if reco['Delta'].max() > 0 else 0
                reco['Exposure_Count'] = reco['Genre'].map(lambda g: director_genre_counts.get(g, 0))
                reco['E'] = 1 / (1 + reco['Exposure_Count'])
                w_E, w_Q, w_T = 0.4, 0.4, 0.2
                reco['Opportunity_Score'] = w_E * reco['E'] + w_Q * reco['Q'] + w_T * reco['T']
                top_recommendations = reco.sort_values(by='Opportunity_Score', ascending=False).head(3)
                st.markdown("**Top genre(s) to consider next:**")
                for _, row in top_recommendations.iterrows():
                    st.write(f"- **{row['Genre']}** (Score: {row['Opportunity_Score']:.3f}) — Global Avg Rating: {row['Overall_Avg_Rating']:.2f}, Momentum Δ: {row['Delta']:.2f}, Past Exposure: {int(row['Exposure_Count'])}")

    if st.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
