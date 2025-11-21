import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Set page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        padding: 0.5rem;
        border-radius: 10px;
    }
    .movie-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .title {
        color: #FF4B4B;
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="title">🎬 Movie Recommender System</p>', unsafe_allow_html=True)
st.markdown("---")

# Function to load data
@st.cache_data
def load_data():
    """Load the movie dataset and similarity matrix"""
    try:
        # Try to load preprocessed data
        movies_df = pd.read_pickle('movies_data.pkl')
        similarity = pickle.load(open('similarity.pkl', 'rb'))
        return movies_df, similarity
    except:
        st.error("⚠️ Data files not found. Please run the preprocessing notebook first.")
        return None, None

# Function to get recommendations
def get_recommendations(movie_title, movies_df, similarity, n_recommendations=5):
    """Get movie recommendations based on similarity"""
    try:
        # Find the movie index
        movie_index = movies_df[movies_df['title'] == movie_title].index[0]
        
        # Get similarity scores
        distances = similarity[movie_index]
        
        # Sort and get top recommendations
        movies_list = sorted(list(enumerate(distances)), 
                           reverse=True, 
                           key=lambda x: x[1])[1:n_recommendations+1]
        
        recommended_movies = []
        for i in movies_list:
            recommended_movies.append({
                'title': movies_df.iloc[i[0]]['title'],
                'similarity': f"{i[1]:.2%}"
            })
        
        return recommended_movies
    
    except IndexError:
        return None

# Main app
def main():
    # Load data
    movies_df, similarity = load_data()
    
    if movies_df is None or similarity is None:
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.write("""
        This movie recommender system uses:
        - **Content-Based Filtering**
        - **Cosine Similarity**
        - **Natural Language Processing**
        
        Select a movie to get personalized recommendations!
        """)
        
        st.markdown("---")
        st.header("⚙️ Settings")
        n_recommendations = st.slider(
            "Number of recommendations",
            min_value=3,
            max_value=10,
            value=5,
            step=1
        )
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Select a Movie")
        
        # Movie selection dropdown
        movie_list = movies_df['title'].values
        selected_movie = st.selectbox(
            "Choose a movie you like:",
            options=movie_list,
            index=0
        )
        
        # Search box as alternative
        st.write("Or search for a movie:")
        search_query = st.text_input("Type movie name...", "")
        
        if search_query:
            filtered_movies = movies_df[
                movies_df['title'].str.contains(search_query, case=False, na=False)
            ]['title'].values
            
            if len(filtered_movies) > 0:
                selected_movie = st.selectbox(
                    "Search results:",
                    options=filtered_movies
                )
    
    with col2:
        st.subheader("Quick Stats")
        st.metric("Total Movies", len(movies_df))
        st.metric("Selected Movie", "✓" if selected_movie else "None")
    
    # Get recommendations button
    if st.button("🎯 Get Recommendations", use_container_width=True):
        with st.spinner("Finding similar movies..."):
            recommendations = get_recommendations(
                selected_movie, 
                movies_df, 
                similarity, 
                n_recommendations
            )
            
            if recommendations:
                st.success(f"✅ Found {len(recommendations)} recommendations!")
                st.markdown("---")
                
                # Display selected movie
                st.subheader("🎬 You selected:")
                st.markdown(f"""
                    <div class="movie-card">
                        <h3>{selected_movie}</h3>
                    </div>
                """, unsafe_allow_html=True)
                
                st.subheader("🎯 Recommended Movies:")
                
                # Display recommendations in columns
                cols = st.columns(2)
                for idx, movie in enumerate(recommendations):
                    with cols[idx % 2]:
                        st.markdown(f"""
                            <div class="movie-card">
                                <h4>{idx + 1}. {movie['title']}</h4>
                                <p>Similarity Score: {movie['similarity']}</p>
                            </div>
                        """, unsafe_allow_html=True)
            else:
                st.error("❌ Movie not found in database. Please try another movie.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; color: gray;'>
            <p>Built with ❤️ using Streamlit | Movie Recommender System</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":

    main()
