import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import string
import re

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("Online_Courses.csv")

df = load_data()

# Preprocess data
df['Skills'] = df['Skills'].fillna('')
df['Course Type'] = df['Course Type'].fillna('')
df['text_features'] = df['Title'] + ' ' + df['Short Intro'] + ' ' + df['Skills'] + ' ' + df['Course Type']

# Function for text preprocessing
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()  # Convert text to lowercase
        text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuations
        return text
    else:
        return ''
    
# Apply text preprocessing
df['text_features'] = df['text_features'].apply(preprocess_text)

# Extract features from text
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
text_features = tfidf_vectorizer.fit_transform(df['text_features'])

# Function to recommend courses based on user input
def recommend_courses(user_input, n=5):
    # Transform user input into TF-IDF vector
    user_vector = tfidf_vectorizer.transform([preprocess_text(user_input)])
    # Calculate similarity between user input and all courses
    similarity_scores = cosine_similarity(user_vector, text_features)
    # Get indices of top n similar courses
    top_indices = similarity_scores.argsort()[0][-n:][::-1]
    # Get recommended courses
    recommended_courses = df.iloc[top_indices]
    return recommended_courses

# Page 1: Course Recommender System
def course_recommender():
    st.title('Course Recommender System')
    
    # Preprocess data
    df['Skills'] = df['Skills'].fillna('')
    df['Course Type'] = df['Course Type'].fillna('')
    df['text_features'] = df['Title'] + ' ' + df['Short Intro'] + ' ' + df['Skills'] + ' ' + df['Course Type']
    df['text_features'] = df['text_features'].apply(preprocess_text)
    
    user_input = st.text_input('Enter your interests, skill level, and career goals:')
    if st.button('Recommend Courses'):
        recommended_courses = recommend_courses(user_input)
        st.write(recommended_courses[['Title', 'URL', 'Short Intro']])

# Page 2: Skill Demand Analysis
def skill_demand_analysis():
    st.title('Skill Demand Analysis')
    
    # Extract skills from the dataset
    skills = df['Skills'].str.split(',').explode().str.strip().dropna()
    
    # Analyze skill frequencies
    skill_counts = Counter(skills)
    
    # Visualize skill demand using a word cloud
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(skill_counts)
    st.subheader('Top In-Demand Skills')
    st.image(wordcloud.to_array())
    
    # Visualize skill demand using a bar chart
    top_skills = skill_counts.most_common(10)
    skills, counts = zip(*top_skills)
    fig, ax = plt.subplots()
    ax.barh(skills, counts, color='skyblue')
    ax.invert_yaxis()  # Invert y-axis to display the most common skill at the top
    ax.set_xlabel('Frequency')
    ax.set_title('Top 10 In-Demand Skills')
    st.pyplot(fig)
    
    # Allow user interaction to explore skill demand by category
    selected_category = st.selectbox('Select a category to explore skill demand:', df['Category'].unique())
    filtered_skills = df[df['Category'] == selected_category]['Skills'].str.split(',').explode().str.strip().dropna()
    filtered_skill_counts = Counter(filtered_skills)
    filtered_top_skills = filtered_skill_counts.most_common(10)
    filtered_skills, filtered_counts = zip(*filtered_top_skills)
    fig_filtered, ax_filtered = plt.subplots()
    ax_filtered.barh(filtered_skills, filtered_counts, color='lightcoral')
    ax_filtered.invert_yaxis()  # Invert y-axis to display the most common skill at the top
    ax_filtered.set_xlabel('Frequency')
    ax_filtered.set_title(f'Top 10 In-Demand Skills in {selected_category}')
    st.pyplot(fig_filtered)

# Page 3: User Engagement Analysis
def user_engagement_analysis():
    st.title('User Engagement Analysis')
    
    # Remove 'stars' from Rating column and convert to float
    df['Rating'] = df['Rating'].str.replace('stars', '').astype(float)
    
    # Extract numerical values from 'Number of viewers' column and convert to float
    def extract_numeric_viewers(viewers):
        numeric_values = []
        for val in viewers:
            # Use regular expression to extract numbers from strings
            numbers = re.findall(r'\d+', str(val))
            if numbers:
                numeric_values.append(float(numbers[0]))
            else:
                numeric_values.append(0)  # Handle cases where no numbers are found
        return numeric_values
    
    df['Number of viewers'] = extract_numeric_viewers(df['Number of viewers'])
    
    # Aggregate view counts per site
    view_counts_per_site = df.groupby('Site')['Number of viewers'].sum()
    
    # Display the aggregated view counts in a table
    st.subheader('View Counts per Site')
    st.write(view_counts_per_site.reset_index().rename(columns={'Number of viewers': 'Total Views'}))
    
    # Top Rated Courses per Site
    st.subheader('Top Rated Courses per Site')
    top_rated_courses_per_site = df.groupby('Site').apply(lambda x: x.nlargest(1, 'Rating')).reset_index(drop=True)
    st.write(top_rated_courses_per_site[['Site', 'Title', 'Rating']])

# Navigation
PAGES = {
    "Course Recommender": course_recommender,
    "Skill Demand Analysis": skill_demand_analysis,
    "User Engagement Analysis": user_engagement_analysis,
}

st.sidebar.title('Enhancing Educational Resource Discovery using Web Mining and Recommender Systems')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page()
