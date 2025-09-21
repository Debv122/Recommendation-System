"""
Streamlit Web Application for Recommendation System
A comprehensive web interface for the getINNOtized recommendation system
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import os
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our recommendation system
from Recommendation_system import RecommendationSystem, clean_and_save_datasets

# Page configuration
st.set_page_config(
    page_title="Recommendation System Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .recommendation-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e0e0e0;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .success-message {
        color: #28a745;
        font-weight: bold;
    }
    .warning-message {
        color: #ffc107;
        font-weight: bold;
    }
    .error-message {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

def load_recommendation_system():
    """Load and initialize the recommendation system"""
    try:
        rs = RecommendationSystem()
        rs.load_data()
        rs.create_user_item_matrix()
        return rs
    except Exception as e:
        st.error(f"Error loading recommendation system: {str(e)}")
        return None

def get_available_users(rs):
    """Get list of available users for recommendations"""
    if rs and rs.user_item_matrix is not None:
        return rs.user_item_matrix.index.tolist()
    return []

def get_system_stats(rs):
    """Get system statistics"""
    if not rs:
        return {}
    
    stats = {}
    try:
        if rs.events_df is not None:
            stats['total_events'] = len(rs.events_df)
            stats['unique_users'] = rs.events_df['visitorid'].nunique()
            stats['unique_items'] = rs.events_df['itemid'].nunique()
            stats['date_range'] = f"{rs.events_df['timestamp'].min().strftime('%Y-%m-%d')} to {rs.events_df['timestamp'].max().strftime('%Y-%m-%d')}"
        
        if rs.user_item_matrix is not None:
            stats['matrix_users'] = len(rs.user_item_matrix)
            stats['matrix_items'] = len(rs.user_item_matrix.columns)
            stats['matrix_sparsity'] = 1 - (rs.user_item_matrix.values != 0).sum() / rs.user_item_matrix.size
        
        if rs.item_properties_df is not None:
            stats['item_properties'] = len(rs.item_properties_df)
            stats['unique_properties'] = rs.item_properties_df['property'].nunique()
            
    except Exception as e:
        st.warning(f"Could not load some statistics: {str(e)}")
    
    return stats

def display_recommendations(recommendations, rs, user_id):
    """Display recommendations in a nice format"""
    if not recommendations:
        st.warning("No recommendations available for this user.")
        return
    
    st.subheader(f"🎯 Recommendations for User {user_id}")
    
    # Create columns for recommendations
    cols = st.columns(min(len(recommendations), 3))
    
    for i, item_id in enumerate(recommendations):
        with cols[i % 3]:
            with st.container():
                st.markdown(f"""
                <div class="recommendation-card">
                    <h4>Item {item_id}</h4>
                    <p><strong>Rank:</strong> {i+1}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Show item details if available
    if rs.item_properties_df is not None and not rs.item_properties_df.empty:
        st.subheader("📋 Item Details")
        item_details = rs.item_properties_df[rs.item_properties_df['itemid'].isin(recommendations)]
        if not item_details.empty:
            st.dataframe(item_details, use_container_width=True)

def create_analytics_dashboard(rs):
    """Create analytics dashboard with visualizations"""
    st.subheader("📊 Analytics Dashboard")
    
    if not rs or rs.events_df is None:
        st.warning("No data available for analytics.")
        return
    
    try:
        # Event distribution
        st.subheader("Event Distribution")
        event_counts = rs.events_df['event'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=event_counts.values, names=event_counts.index, 
                         title="Event Type Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(x=event_counts.index, y=event_counts.values,
                         title="Event Counts by Type")
            st.plotly_chart(fig, use_container_width=True)
        
        # Temporal analysis
        st.subheader("Temporal Analysis")
        
        # Hourly activity
        rs.events_df['hour'] = rs.events_df['timestamp'].dt.hour
        hourly_activity = rs.events_df.groupby('hour').size()
        
        fig = px.line(x=hourly_activity.index, y=hourly_activity.values,
                      title="Activity by Hour of Day",
                      labels={'x': 'Hour', 'y': 'Number of Events'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Daily activity
        rs.events_df['day_of_week'] = rs.events_df['timestamp'].dt.day_name()
        daily_activity = rs.events_df.groupby('day_of_week').size()
        
        fig = px.bar(x=daily_activity.index, y=daily_activity.values,
                     title="Activity by Day of Week")
        st.plotly_chart(fig, use_container_width=True)
        
        # User activity analysis
        st.subheader("User Activity Analysis")
        user_activity = rs.events_df.groupby('visitorid').size()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Average Events per User", f"{user_activity.mean():.1f}")
            st.metric("Median Events per User", f"{user_activity.median():.1f}")
        
        with col2:
            st.metric("Most Active User", f"{user_activity.idxmax()}")
            st.metric("Max Events by User", f"{user_activity.max()}")
        
        # Item popularity
        st.subheader("Item Popularity")
        item_popularity = rs.events_df[rs.events_df['event'] == 'view']['itemid'].value_counts().head(20)
        
        fig = px.bar(x=item_popularity.values, y=item_popularity.index,
                     orientation='h', title="Top 20 Most Viewed Items")
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating analytics dashboard: {str(e)}")
        st.info("Some visualizations may not be available due to data limitations.")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Recommendation System Dashboard</h1>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["🏠 Home", "🎯 Recommendations", "📊 Analytics", "⚙️ System Status", "🔧 Data Management"]
    )
    
    # Add cache clearing option in sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Cache & Reload"):
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    
    # Initialize session state
    if 'rs' not in st.session_state:
        st.session_state.rs = None
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'system_stats' not in st.session_state:
        st.session_state.system_stats = {}
    if 'available_users' not in st.session_state:
        st.session_state.available_users = []
    
    # Home page
    if page == "🏠 Home":
        st.markdown("""
        ## Welcome to the Recommendation System Dashboard
        
        This application provides a comprehensive interface for the getINNOtized recommendation system.
        
        ### Features:
        - **Personalized Recommendations**: Get item recommendations for any user
        - **Multiple Algorithms**: Collaborative filtering, content-based, and hybrid approaches
        - **Analytics Dashboard**: Visualize user behavior and system performance
        - **Real-time Processing**: Dynamic recommendation generation
        - **System Monitoring**: Track system health and performance metrics
        
        ### Getting Started:
        1. Navigate to "System Status" to ensure data is loaded
        2. Go to "Recommendations" to get personalized suggestions
        3. Explore "Analytics" for insights into user behavior
        """)
        
        # Quick stats
        if st.button("🔄 Load System Data", type="primary"):
            with st.spinner("Loading recommendation system..."):
                try:
                    st.session_state.rs = load_recommendation_system()
                    if st.session_state.rs:
                        st.session_state.data_loaded = True
                        # Update cached data
                        st.session_state.system_stats = get_system_stats(st.session_state.rs)
                        st.session_state.available_users = get_available_users(st.session_state.rs)
                        st.success("✅ System loaded successfully!")
                    else:
                        st.error("❌ Failed to load system data.")
                except Exception as e:
                    st.error(f"❌ Error loading system: {str(e)}")
        
        if st.session_state.data_loaded and st.session_state.rs:
            stats = st.session_state.system_stats
            
            st.subheader("📈 System Overview")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Events", f"{stats.get('total_events', 0):,}")
            with col2:
                st.metric("Unique Users", f"{stats.get('unique_users', 0):,}")
            with col3:
                st.metric("Unique Items", f"{stats.get('unique_items', 0):,}")
            with col4:
                st.metric("Matrix Sparsity", f"{stats.get('matrix_sparsity', 0):.3f}")
    
    # Recommendations page
    elif page == "🎯 Recommendations":
        st.header("🎯 Personalized Recommendations")
        
        if not st.session_state.data_loaded or not st.session_state.rs:
            st.warning("⚠️ Please load system data first from the Home page.")
            return
        
        rs = st.session_state.rs
        
        # User selection
        st.subheader("Select User")
        available_users = st.session_state.available_users
        
        if not available_users:
            st.error("No users available in the system.")
            return
        
        user_id = st.selectbox(
            "Choose a user ID:",
            available_users,
            help="Select a user to get personalized recommendations"
        )
        
        # Recommendation parameters
        st.subheader("Recommendation Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            n_recommendations = st.slider(
                "Number of recommendations:",
                min_value=1,
                max_value=20,
                value=5,
                help="How many items to recommend"
            )
        
        with col2:
            algorithm = st.selectbox(
                "Recommendation Algorithm:",
                ["Hybrid", "Collaborative Filtering", "Content-Based", "Matrix Factorization", "Popularity-Based"],
                help="Choose the recommendation algorithm"
            )
        
        # Get recommendations
        if st.button("🚀 Get Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                try:
                    if algorithm == "Hybrid":
                        recommendations = rs.hybrid_recommendations(user_id, n_recommendations)
                    elif algorithm == "Collaborative Filtering":
                        recommendations = rs.collaborative_filtering_recommendations(user_id, n_recommendations)
                    elif algorithm == "Content-Based":
                        recommendations = rs.content_based_recommendations(user_id, n_recommendations)
                    elif algorithm == "Matrix Factorization":
                        recommendations = rs.matrix_factorization_recommendations(user_id, n_recommendations)
                    elif algorithm == "Popularity-Based":
                        recommendations = rs.popularity_based_recommendations(n_recommendations)
                    
                    if recommendations:
                        display_recommendations(recommendations, rs, user_id)
                        
                        # Show user's interaction history
                        st.subheader("👤 User Interaction History")
                        user_items = rs.user_item_matrix.loc[user_id]
                        user_interacted_items = user_items[user_items > 0].index.tolist()
                        
                        if user_interacted_items:
                            st.write(f"User has interacted with {len(user_interacted_items)} items:")
                            st.dataframe(pd.DataFrame({'Item ID': user_interacted_items[:10]}), use_container_width=True)
                        else:
                            st.info("This user has no interaction history.")
                    
                except Exception as e:
                    st.error(f"Error generating recommendations: {str(e)}")
    
    # Analytics page
    elif page == "📊 Analytics":
        st.header("📊 Analytics Dashboard")
        
        if not st.session_state.data_loaded or not st.session_state.rs:
            st.warning("⚠️ Please load system data first from the Home page.")
            return
        
        create_analytics_dashboard(st.session_state.rs)
    
    # System Status page
    elif page == "⚙️ System Status":
        st.header("⚙️ System Status")
        
        if st.session_state.data_loaded and st.session_state.rs:
            st.success("✅ System is loaded and ready!")
            
            # System statistics
            stats = st.session_state.system_stats
            
            st.subheader("📊 System Statistics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Events", f"{stats.get('total_events', 0):,}")
                st.metric("Unique Users", f"{stats.get('unique_users', 0):,}")
                st.metric("Unique Items", f"{stats.get('unique_items', 0):,}")
                st.metric("Date Range", stats.get('date_range', 'N/A'))
            
            with col2:
                st.metric("Matrix Users", f"{stats.get('matrix_users', 0):,}")
                st.metric("Matrix Items", f"{stats.get('matrix_items', 0):,}")
                st.metric("Matrix Sparsity", f"{stats.get('matrix_sparsity', 0):.3f}")
                st.metric("Item Properties", f"{stats.get('item_properties', 0):,}")
            
            # Data quality metrics
            st.subheader("🔍 Data Quality")
            
            if st.session_state.rs.user_item_matrix is not None:
                matrix = st.session_state.rs.user_item_matrix
                users_with_interactions = (matrix.sum(axis=1) > 0).sum()
                items_with_interactions = (matrix.sum(axis=0) > 0).sum()
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Users with Interactions", users_with_interactions)
                with col2:
                    st.metric("Items with Interactions", items_with_interactions)
                with col3:
                    st.metric("Total Interactions", int(matrix.sum().sum()))
        
        else:
            st.warning("⚠️ System not loaded. Please go to the Home page to load data.")
    
    # Data Management page
    elif page == "🔧 Data Management":
        st.header("🔧 Data Management")
        
        st.subheader("Data Cleaning")
        st.write("Clean and preprocess the raw datasets to create optimized versions.")
        
        if st.button("🧹 Clean and Save Datasets", type="primary"):
            with st.spinner("Cleaning datasets..."):
                try:
                    clean_and_save_datasets()
                    st.success("✅ Datasets cleaned and saved successfully!")
                    st.info("Cleaned datasets are now available for use.")
                except Exception as e:
                    st.error(f"❌ Error cleaning datasets: {str(e)}")
        
        st.subheader("File Status")
        
        # Check file existence
        files_to_check = [
            "events_cleaned.csv",
            "item_properties_cleaned.csv", 
            "category_tree_cleaned.csv"
        ]
        
        for file in files_to_check:
            if os.path.exists(file):
                file_size = os.path.getsize(file) / (1024 * 1024)  # MB
                st.success(f"✅ {file} ({file_size:.1f} MB)")
            else:
                st.warning(f"⚠️ {file} not found")
        
        st.subheader("System Configuration")
        st.write("Current system configuration:")
        
        config_data = {
            "MAX_ROWS_EVENTS": os.getenv('MAX_ROWS_EVENTS', '300000'),
            "MAX_ROWS_ITEM_PROPERTIES": os.getenv('MAX_ROWS_ITEM_PROPERTIES', '200000'),
            "TOP_USERS_LIMIT": os.getenv('TOP_USERS_LIMIT', '1000'),
            "TOP_ITEMS_LIMIT": os.getenv('TOP_ITEMS_LIMIT', '1000'),
            "COLLAB_SIMILAR_USERS": os.getenv('COLLAB_SIMILAR_USERS', '10')
        }
        
        st.dataframe(pd.DataFrame(list(config_data.items()), columns=['Parameter', 'Value']), use_container_width=True)

if __name__ == "__main__":
    main()
