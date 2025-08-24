"""
Recommendation System for getINNOtized
A comprehensive recommendation system leveraging user behavior and preferences
"""

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import os
import warnings
warnings.filterwarnings('ignore')

# Runtime/memory configuration (tunable via env vars; safe defaults for Colab)
MAX_ROWS_EVENTS = int(os.getenv('MAX_ROWS_EVENTS', '300000'))
MAX_ROWS_ITEM_PROPERTIES = int(os.getenv('MAX_ROWS_ITEM_PROPERTIES', '200000'))
EVENT_COLUMNS = ['timestamp', 'visitorid', 'event', 'itemid', 'transactionid']
TOP_USERS_LIMIT = int(os.getenv('TOP_USERS_LIMIT', '1000'))  # Increased from 300
TOP_ITEMS_LIMIT = int(os.getenv('TOP_ITEMS_LIMIT', '1000'))  # Increased from 300
COLLAB_SIMILAR_USERS = int(os.getenv('COLLAB_SIMILAR_USERS', '10'))  # Increased from 5
CONTENT_BASED_ITEM_LIMIT = int(os.getenv('CONTENT_BASED_ITEM_LIMIT', '10000'))  # Increased from 5000
N_TEST_USERS = int(os.getenv('N_TEST_USERS', '100'))  # Increased from 50
EVAL_TOP_N = int(os.getenv('EVAL_TOP_N', '10'))  # Increased from 5
EVAL_USER_LIMIT = int(os.getenv('EVAL_USER_LIMIT', '50'))  # Increased from 30
SKIP_HEAVY_ANALYTICS = os.getenv('SKIP_HEAVY_ANALYTICS', '1') == '1'

# Ensure UTF-8 console output on Windows to prevent UnicodeEncodeError in prints
try:
    import sys
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

class RecommendationSystem:
    def __init__(self):
        self.events_df = None
        self.category_tree_df = None
        self.item_properties_df = None
        self.user_item_matrix = None
        self.item_similarity_matrix = None
        self.user_similarity_matrix = None
        
    def load_data(self):
        """Load and preprocess all data files"""
        print("Loading data files...")
        
        # Prefer cleaned datasets if available
        events_path = 'events_cleaned.csv' if os.path.exists('events_cleaned.csv') else 'events.csv'
        cat_path = 'category_tree_cleaned.csv' if os.path.exists('category_tree_cleaned.csv') else 'category_tree.csv'
        
        # Load events data with sampling for large files
        print("Loading events data...")
        try:
            # Try to load with row cap and selected columns
            self.events_df = pd.read_csv(events_path, usecols=[c for c in EVENT_COLUMNS if c != 'transactionid' or 'transactionid' in pd.read_csv(events_path, nrows=1).columns], nrows=MAX_ROWS_EVENTS)
        except MemoryError:
            print("Memory error, loading smaller sample of events data...")
            self.events_df = pd.read_csv(events_path, usecols=[c for c in EVENT_COLUMNS if c != 'transactionid' or 'transactionid' in pd.read_csv(events_path, nrows=1).columns], nrows=max(50000, MAX_ROWS_EVENTS // 2))
        
        print(f"Events data shape: {self.events_df.shape}")
        
        # Load category tree
        self.category_tree_df = pd.read_csv(cat_path)
        print(f"Category tree shape: {self.category_tree_df.shape}")
        
        # Load item properties (prefer cleaned file, else combine known parts)
        if os.path.exists('item_properties_cleaned.csv'):
            self.item_properties_df = pd.read_csv('item_properties_cleaned.csv', usecols=['itemid', 'property', 'value'], nrows=MAX_ROWS_ITEM_PROPERTIES)
            print(f"Item properties shape: {self.item_properties_df.shape}")
        else:
            try:
                if os.path.exists('item_properties.csv'):
                    self.item_properties_df = pd.read_csv('item_properties.csv', usecols=['itemid', 'property', 'value'], nrows=MAX_ROWS_ITEM_PROPERTIES)
                else:
                    frames = []
                    remaining = MAX_ROWS_ITEM_PROPERTIES
                    if os.path.exists('item_properties_part1.1.csv') and remaining > 0:
                        n1 = remaining // 2 if os.path.exists('item_properties_part2.csv') else remaining
                        frames.append(pd.read_csv('item_properties_part1.1.csv', usecols=['itemid', 'property', 'value'], nrows=n1))
                        remaining -= n1
                    if os.path.exists('item_properties_part2.csv') and remaining > 0:
                        frames.append(pd.read_csv('item_properties_part2.csv', usecols=['itemid', 'property', 'value'], nrows=remaining))
                    self.item_properties_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                print(f"Item properties shape: {self.item_properties_df.shape}")
            except Exception:
                print("Could not load item properties, using empty DataFrame")
                self.item_properties_df = pd.DataFrame()
        
        # Convert timestamp to datetime
        self.events_df['timestamp'] = pd.to_datetime(self.events_df['timestamp'], unit='ms', errors='coerce')
        self.events_df = self.events_df.dropna(subset=['timestamp'])
        
        print("Data loading completed!")
        
    def create_user_item_matrix(self):
        """Create user-item interaction matrix"""
        print("Creating user-item matrix...")
        
        # Create interaction matrix (view events)
        view_events = self.events_df[self.events_df['event'] == 'view']
        print(f"Total view events: {len(view_events)}")
        print(f"Unique users in view events: {view_events['visitorid'].nunique()}")
        print(f"Unique items in view events: {view_events['itemid'].nunique()}")
        
        # Sample data for memory efficiency (use top users and items)
        print("Sampling data for memory efficiency...")
        
        # Get top users by activity
        user_activity = view_events.groupby('visitorid').size()
        top_users = user_activity.nlargest(TOP_USERS_LIMIT).index.tolist()
        print(f"Top {len(top_users)} users selected")
        print(f"Sample top users: {top_users[:5]}")
        
        # Get top items by popularity
        item_popularity = view_events.groupby('itemid').size()
        top_items = item_popularity.nlargest(TOP_ITEMS_LIMIT).index.tolist()
        print(f"Top {len(top_items)} items selected")
        print(f"Sample top items: {top_items[:5]}")
        
        # Filter to top users and items
        filtered_events = view_events[
            (view_events['visitorid'].isin(top_users)) &
            (view_events['itemid'].isin(top_items))
        ]
        print(f"Filtered events: {len(filtered_events)}")
        
        # Create user-item matrix with interaction counts
        self.user_item_matrix = filtered_events.groupby(['visitorid', 'itemid']).size().unstack(fill_value=0)
        
        print(f"User-item matrix shape: {self.user_item_matrix.shape}")
        print(f"Matrix sparsity: {1 - (self.user_item_matrix.values != 0).sum() / self.user_item_matrix.size:.3f}")
        print(f"Sample matrix values:")
        print(self.user_item_matrix.head(3).iloc[:, :5])
        
        # Check for users with interactions
        users_with_interactions = (self.user_item_matrix.sum(axis=1) > 0).sum()
        print(f"Users with interactions: {users_with_interactions}")
        
        return self.user_item_matrix
    
    def collaborative_filtering_recommendations(self, user_id, n_recommendations=5):
        """Collaborative filtering based recommendations with improved sparse data handling"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        try:
            # Calculate user similarity
            user_similarities = cosine_similarity([self.user_item_matrix.loc[user_id]])
            
            # Find similar users (increase the number to handle sparsity)
            similar_users = np.argsort(user_similarities[0])[::-1][1:COLLAB_SIMILAR_USERS*3+1]  # Get even more similar users
            
            # Get items liked by similar users with diversity
            recommendations = []
            used_items = set()
            
            for similar_user_idx in similar_users:
                similar_user_id = self.user_item_matrix.index[similar_user_idx]
                user_items = self.user_item_matrix.loc[similar_user_id]
                liked_items = user_items[user_items > 0].index.tolist()
                
                # Add items we haven't seen yet
                for item in liked_items:
                    if item not in used_items:
                        recommendations.append(item)
                        used_items.add(item)
                        if len(recommendations) >= n_recommendations * 2:  # Get more candidates
                            break
                
                if len(recommendations) >= n_recommendations * 2:
                    break
            
            # Remove items user already interacted with
            user_items = self.user_item_matrix.loc[user_id]
            user_interacted_items = user_items[user_items > 0].index.tolist()
            recommendations = [item for item in recommendations if item not in user_interacted_items]
            
            # If we don't have enough recommendations, add popular items
            if len(recommendations) < n_recommendations:
                popular_items = self.popularity_based_recommendations(n_recommendations * 2)
                for item in popular_items:
                    if item not in recommendations and item not in user_interacted_items:
                        recommendations.append(item)
                        if len(recommendations) >= n_recommendations:
                            break
            
            # Shuffle recommendations to add diversity
            import random
            random.shuffle(recommendations)
            
            # Return top N recommendations
            return list(dict.fromkeys(recommendations))[:n_recommendations]
            
        except Exception as e:
            print(f"Collaborative filtering failed: {e}")
            return []
    
    def content_based_recommendations(self, user_id, n_recommendations=5):
        """Content-based recommendations using one-hot encoded categorical features"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's interacted items
        user_items = self.user_item_matrix.loc[user_id]
        user_interacted_items = user_items[user_items > 0].index.tolist()
        
        if not user_interacted_items:
            return []
        
        # One-hot encode all categorical ID features
        cat_onehot = one_hot_encode_categorical_ids(self.item_properties_df)
        if cat_onehot.empty:
            print("No categorical features available for content-based recommendations.")
            return []
        
        # Build user profile as mean of one-hot vectors for interacted items
        user_item_vectors = cat_onehot.loc[cat_onehot.index.intersection(user_interacted_items)]
        if user_item_vectors.empty:
            return []
        
        user_profile = user_item_vectors.mean(axis=0).values.reshape(1, -1)
        
        # Candidate items: those not already interacted with, and present in cat_onehot
        candidate_items = cat_onehot.index.difference(user_interacted_items)
        if candidate_items.empty:
            return []
        
        candidate_vectors = cat_onehot.loc[candidate_items]
        
        # Compute cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity(user_profile, candidate_vectors.values)[0]
        
        # Get top-N recommendations
        top_indices = similarities.argsort()[::-1][:n_recommendations*2]  # Get more candidates
        recommended_items = candidate_vectors.index[top_indices].tolist()
        
        # If we don't have enough, add some popular items as fallback
        if len(recommended_items) < n_recommendations:
            popular_items = self.popularity_based_recommendations(n_recommendations * 2)
            for item in popular_items:
                if item not in recommended_items and item not in user_interacted_items:
                    recommended_items.append(item)
                    if len(recommended_items) >= n_recommendations:
                        break
        
        return recommended_items[:n_recommendations]
    
    def popularity_based_recommendations(self, n_recommendations=5):
        """Popularity-based recommendations"""
        # Count item interactions
        item_popularity = self.events_df[self.events_df['event'] == 'view']['itemid'].value_counts()
        return item_popularity.head(n_recommendations).index.tolist()
    
    def category_based_recommendations(self, user_id, n_recommendations=5):
        """Category-based recommendations"""
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's interacted items
        user_items = self.user_item_matrix.loc[user_id]
        user_interacted_items = user_items[user_items > 0].index.tolist()
        
        if not user_interacted_items:
            return []
        
        # Get categories of user's items
        user_item_categories = self.item_properties_df[
            (self.item_properties_df['itemid'].isin(user_interacted_items)) &
            (self.item_properties_df['property'] == 'categoryid')
        ]['value'].unique()
        
        # Find items in same categories
        category_items = self.item_properties_df[
            (self.item_properties_df['property'] == 'categoryid') &
            (self.item_properties_df['value'].isin(user_item_categories))
        ]['itemid'].unique()
        
        # Remove items user already interacted with
        recommendations = [item for item in category_items if item not in user_interacted_items]
        
        return recommendations[:n_recommendations]
    
    def hybrid_recommendations(self, user_id, n_recommendations=5):
        """Hybrid recommendations combining multiple approaches"""
        print(f"DEBUG: Generating hybrid recommendations for user {user_id}")
        
        collab_recs = self.collaborative_filtering_recommendations(user_id, n_recommendations)
        content_recs = self.content_based_recommendations(user_id, n_recommendations)
        category_recs = self.category_based_recommendations(user_id, n_recommendations)
        
        print(f"  Collaborative recommendations: {len(collab_recs)} items")
        print(f"  Content-based recommendations: {len(content_recs)} items")
        print(f"  Category-based recommendations: {len(category_recs)} items")
        
        # Combine and weight recommendations
        all_recs = collab_recs + content_recs + category_recs
        rec_counts = {}
        
        for rec in all_recs:
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        
        # Sort by frequency and return top N
        sorted_recs = sorted(rec_counts.items(), key=lambda x: x[1], reverse=True)
        final_recs = [item_id for item_id, _ in sorted_recs[:n_recommendations]]
        
        print(f"  Combined unique recommendations: {len(rec_counts)} items")
        print(f"  Final recommendations: {len(final_recs)} items")
        
        return final_recs
    
    def strong_hybrid_recommendations(self, user_id, n_recommendations=5, weights=None):
        """Stronger hybrid recommendations: weighted sum of normalized collaborative, content, and popularity scores."""
        if weights is None:
            weights = {'collaborative': 0.5, 'content': 0.3, 'popularity': 0.2}
        # Get candidate items from all methods
        collab_recs = self.collaborative_filtering_recommendations(user_id, n_recommendations*5)
        content_recs = self.content_based_recommendations(user_id, n_recommendations*5)
        pop_recs = self.popularity_based_recommendations(n_recommendations*5)
        
        # Union of all candidate items
        candidate_items = set(collab_recs + content_recs + pop_recs)
        if user_id in self.user_item_matrix.index:
            user_items = self.user_item_matrix.loc[user_id]
            user_interacted_items = set(user_items[user_items > 0].index.tolist())
        else:
            user_interacted_items = set()
        candidate_items = [item for item in candidate_items if item not in user_interacted_items]
        if not candidate_items:
            return []
        # Collaborative score
        collab_scores = {item: 0 for item in candidate_items}
        for rank, item in enumerate(collab_recs):
            if item in collab_scores:
                collab_scores[item] = len(collab_recs) - rank
        # Content score
        content_scores = {item: 0 for item in candidate_items}
        for rank, item in enumerate(content_recs):
            if item in content_scores:
                content_scores[item] = len(content_recs) - rank
        # Popularity score
        pop_scores = {item: 0 for item in candidate_items}
        for rank, item in enumerate(pop_recs):
            if item in pop_scores:
                pop_scores[item] = len(pop_recs) - rank
        # Normalize scores
        def normalize(scores):
            vals = list(scores.values())
            min_v, max_v = min(vals), max(vals)
            if max_v == min_v:
                return {k: 0 for k in scores}
            return {k: (v - min_v) / (max_v - min_v) for k, v in scores.items()}
        collab_norm = normalize(collab_scores)
        content_norm = normalize(content_scores)
        pop_norm = normalize(pop_scores)
        # Weighted sum
        final_scores = {}
        for item in candidate_items:
            final_scores[item] = (
                weights['collaborative'] * collab_norm.get(item, 0) +
                weights['content'] * content_norm.get(item, 0) +
                weights['popularity'] * pop_norm.get(item, 0)
            )
        # Sort and return top N
        sorted_items = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items[:n_recommendations]]
    
    def simple_fallback_recommendations(self, user_id, n_recommendations=5):
        """Simple fallback recommendations to ensure we always have some suggestions"""
        # Get all items in the system
        all_items = set(self.user_item_matrix.columns)
        
        # Get items user has already interacted with
        user_items = self.user_item_matrix.loc[user_id]
        user_interacted_items = set(user_items[user_items > 0].index)
        
        # Get popular items that user hasn't interacted with
        available_items = all_items - user_interacted_items
        
        if not available_items:
            return []
        
        # Get item popularity (total interactions across all users)
        item_popularity = self.user_item_matrix.sum(axis=0)
        popular_items = item_popularity[list(available_items)].sort_values(ascending=False)
        
        # Return top N popular items
        return popular_items.head(n_recommendations).index.tolist()

    def matrix_factorization_recommendations(self, user_id, n_recommendations=5):
        """Matrix factorization recommendations using SVD - works better with sparse data"""
        try:
            from sklearn.decomposition import TruncatedSVD
            
            # Get user index
            if user_id not in self.user_item_matrix.index:
                return []
            
            # Prepare matrix for SVD (fill NaN with 0)
            matrix = self.user_item_matrix.fillna(0).values
            
            # Apply SVD to the user-item matrix
            n_components = min(10, min(matrix.shape)-1)
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            
            # Transform the matrix: users × components
            user_factors = svd.fit_transform(matrix)
            # Transform back: components × items  
            item_factors = svd.components_.T
            
            # Find user in reduced space
            user_idx = self.user_item_matrix.index.get_loc(user_id)
            user_vector = user_factors[user_idx]
            
            # Calculate similarities with all items
            item_similarities = []
            for item_idx in range(len(item_factors)):
                item_vector = item_factors[item_idx]
                similarity = np.dot(user_vector, item_vector)
                item_similarities.append((item_idx, similarity))
            
            # Sort by similarity and get top N
            item_similarities.sort(key=lambda x: x[1], reverse=True)
            top_indices = [idx for idx, _ in item_similarities[:n_recommendations*2]]  # Get more candidates
            
            # Get item IDs
            recommended_items = self.user_item_matrix.columns[top_indices].tolist()
            
            # Filter out items user already interacted with
            user_items = self.user_item_matrix.loc[user_id]
            user_interacted_items = set(user_items[user_items > 0].index)
            recommended_items = [item for item in recommended_items if item not in user_interacted_items]
            
            return recommended_items[:n_recommendations]
            
        except Exception as e:
            print(f"Matrix factorization failed: {e}")
            return []

    def robust_hybrid_recommendations(self, user_id, n_recommendations=5):
        """Robust hybrid recommendations with multiple fallbacks"""
        # Try the main hybrid method first
        recommendations = self.hybrid_recommendations(user_id, n_recommendations)
        
        # If no recommendations, try matrix factorization
        if not recommendations:
            print(f"  Main hybrid failed, trying matrix factorization for user {user_id}")
            recommendations = self.matrix_factorization_recommendations(user_id, n_recommendations)
        
        # If still no recommendations, use fallback
        if not recommendations:
            print(f"  Matrix factorization failed, using fallback for user {user_id}")
            recommendations = self.simple_fallback_recommendations(user_id, n_recommendations)
        
        return recommendations
    
    def analyze_user_behavior(self):
        """Analyze user behavior patterns"""
        print("\n=== User Behavior Analysis ===")
        
        # Event distribution
        event_counts = self.events_df['event'].value_counts()
        print(f"Event distribution:\n{event_counts}")
        
        # User activity analysis
        user_activity = self.events_df.groupby('visitorid').size()
        print(f"\nUser activity statistics:")
        print(f"Average events per user: {user_activity.mean():.2f}")
        print(f"Median events per user: {user_activity.median():.2f}")
        print(f"Most active user: {user_activity.idxmax()} with {user_activity.max()} events")
        
        # Item popularity analysis
        item_popularity = self.events_df[self.events_df['event'] == 'view']['itemid'].value_counts()
        print(f"\nItem popularity statistics:")
        print(f"Most viewed item: {item_popularity.index[0]} with {item_popularity.iloc[0]} views")
        print(f"Average views per item: {item_popularity.mean():.2f}")
        
        return {
            'event_counts': event_counts,
            'user_activity': user_activity,
            'item_popularity': item_popularity
        }
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in user behavior"""
        print("\n=== Temporal Pattern Analysis ===")
        
        # Hourly activity
        self.events_df['hour'] = self.events_df['timestamp'].dt.hour
        hourly_activity = self.events_df.groupby('hour').size()
        
        print(f"Peak activity hour: {hourly_activity.idxmax()} ({hourly_activity.max()} events)")
        print(f"Lowest activity hour: {hourly_activity.idxmin()} ({hourly_activity.min()} events)")
        
        # Daily activity
        self.events_df['day_of_week'] = self.events_df['timestamp'].dt.day_name()
        daily_activity = self.events_df.groupby('day_of_week').size()
        
        print(f"\nMost active day: {daily_activity.idxmax()} ({daily_activity.max()} events)")
        print(f"Least active day: {daily_activity.idxmin()} ({daily_activity.min()} events)")
        
        return {
            'hourly_activity': hourly_activity,
            'daily_activity': daily_activity
        }
    
    def evaluate_recommendations(self, test_users=None, n_test_users=100, top_n=10, min_interactions=3, eval_user_limit=100, test_ratio=0.2, debug_level=1):
        """Evaluate recommendation quality with proper train-test split
        - Creates a train-test split by holding out some user interactions
        - Computes average precision, recall, and F1 across evaluated users
        - Computes Top-N Accuracy (HitRate@N): fraction of users with at least one relevant item in top-N
        
        debug_level: 0=minimal, 1=summary per user, 2=detailed per user
        """
        if test_users is None:
            # Sample users for evaluation
            active_users = self.user_item_matrix.sum(axis=1).sort_values(ascending=False)
            test_users = active_users.head(n_test_users).index.tolist()
        
        print(f"\n=== Recommendation Evaluation (with Train-Test Split) ===")
        print(f"Evaluating recommendations for {len(test_users)} users")
        print(f"Test ratio: {test_ratio} ({int(test_ratio*100)}% of interactions held out for testing)")
        
        # Calculate metrics
        precision_scores = []
        recall_scores = []
        hits_topn = 0
        evaluated_users = 0
        
        for user_id in test_users:
            # Get actual user interactions
            user_items_series = self.user_item_matrix.loc[user_id]
            user_interactions = set(user_items_series[user_items_series > 0].index)
            
            if len(user_interactions) < min_interactions:
                continue
            
            # Create train-test split for this user
            user_interactions_list = list(user_interactions)
            test_size = max(1, int(len(user_interactions_list) * test_ratio))
            
            # Hold out some interactions for testing (ground truth)
            test_items = set(user_interactions_list[:test_size])
            train_items = set(user_interactions_list[test_size:])
            
            # Temporarily modify user_item_matrix to exclude test items for training
            original_values = self.user_item_matrix.loc[user_id, list(test_items)].copy()
            self.user_item_matrix.loc[user_id, list(test_items)] = 0
            
            # Get recommendations based on training data only
            recommendations = set(self.robust_hybrid_recommendations(user_id, top_n))
            
            # Restore original values
            self.user_item_matrix.loc[user_id, list(test_items)] = original_values
            
            # DEBUG: Print user information
            if debug_level >= 1:
                print(f"\nDEBUG - User {user_id}:")
                print(f"  Total interactions: {len(user_interactions)} items")
                print(f"  Training items: {len(train_items)} items")
                print(f"  Test items (ground truth): {len(test_items)} items")
                print(f"  Recommendations generated: {len(recommendations)} items")
                print(f"  Sample test items: {list(test_items)[:3]}")
                print(f"  Sample recommendations: {list(recommendations)[:3]}")
            
            # Only evaluate users for whom we produced recommendations
            if recommendations:
                # Calculate overlap between recommendations and test items (ground truth)
                relevant_recommended = len(test_items & recommendations)
                precision = relevant_recommended / len(recommendations) if len(recommendations) else 0
                recall = relevant_recommended / len(test_items) if len(test_items) else 0
                
                if debug_level >= 1:
                    print(f"  Overlap (relevant recommended): {relevant_recommended}")
                    print(f"  Precision: {precision:.3f}")
                    print(f"  Recall: {recall:.3f}")
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                
                if relevant_recommended > 0:
                    hits_topn += 1
                
                evaluated_users += 1
                
                if evaluated_users >= eval_user_limit:
                    break
            else:
                if debug_level >= 1:
                    print(f"  No recommendations generated for this user!")
        
        avg_precision = float(np.mean(precision_scores)) if precision_scores else 0.0
        avg_recall = float(np.mean(recall_scores)) if recall_scores else 0.0
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        topn_accuracy = (hits_topn / evaluated_users) if evaluated_users > 0 else 0.0
        
        print(f"\n=== Final Results ===")
        print(f"Users evaluated: {evaluated_users}")
        print(f"Users with hits: {hits_topn}")
        print(f"Average Precision: {avg_precision:.3f}")
        print(f"Average Recall: {avg_recall:.3f}")
        print(f"F1 Score: {f1_score:.3f}")
        print(f"Top-{top_n} Accuracy (HitRate@{top_n}): {topn_accuracy:.3f} ({hits_topn}/{evaluated_users} users)")
        
        # Show distribution of precision/recall scores
        if debug_level >= 1 and precision_scores:
            print(f"\n=== Score Distribution ===")
            print(f"Precision scores: {len([p for p in precision_scores if p > 0])} users > 0, {len([p for p in precision_scores if p == 0])} users = 0")
            print(f"Recall scores: {len([r for r in recall_scores if r > 0])} users > 0, {len([r for r in recall_scores if r == 0])} users = 0")
            print(f"Best precision: {max(precision_scores):.3f}, Best recall: {max(recall_scores):.3f}")
            print(f"Worst precision: {min(precision_scores):.3f}, Worst recall: {min(recall_scores):.3f}")
        
        # --- Evaluate strong hybrid ---
        print(f"\n=== Strong Hybrid Recommendation Evaluation ===")
        precision_scores2 = []
        recall_scores2 = []
        hits_topn2 = 0
        evaluated_users2 = 0
        
        for user_id in test_users:
            user_items_series = self.user_item_matrix.loc[user_id]
            user_interactions = set(user_items_series[user_items_series > 0].index)
            if len(user_interactions) < min_interactions:
                continue
            
            # Create train-test split for this user
            user_interactions_list = list(user_interactions)
            test_size = max(1, int(len(user_interactions_list) * test_ratio))
            test_items = set(user_interactions_list[:test_size])
            train_items = set(user_interactions_list[test_size:])
            
            # Temporarily modify user_item_matrix to exclude test items for training
            original_values = self.user_item_matrix.loc[user_id, list(test_items)].copy()
            self.user_item_matrix.loc[user_id, list(test_items)] = 0
            
            # Get recommendations based on training data only
            recommendations2 = set(self.strong_hybrid_recommendations(user_id, top_n))
            
            # Restore original values
            self.user_item_matrix.loc[user_id, list(test_items)] = original_values
            
            if recommendations2:
                relevant_recommended2 = len(test_items & recommendations2)
                precision2 = relevant_recommended2 / len(recommendations2) if len(recommendations2) else 0
                recall2 = relevant_recommended2 / len(test_items) if len(test_items) else 0
                precision_scores2.append(precision2)
                recall_scores2.append(recall2)
                if relevant_recommended2 > 0:
                    hits_topn2 += 1
                evaluated_users2 += 1
                if evaluated_users2 >= eval_user_limit:
                    break
        
        avg_precision2 = float(np.mean(precision_scores2)) if precision_scores2 else 0.0
        avg_recall2 = float(np.mean(recall_scores2)) if recall_scores2 else 0.0
        f1_score2 = 2 * (avg_precision2 * avg_recall2) / (avg_precision2 + avg_recall2) if (avg_precision2 + avg_recall2) > 0 else 0.0
        topn_accuracy2 = (hits_topn2 / evaluated_users2) if evaluated_users2 > 0 else 0.0
        
        print(f"[Strong Hybrid] Average Precision: {avg_precision2:.3f}")
        print(f"[Strong Hybrid] Average Recall: {avg_recall2:.3f}")
        print(f"[Strong Hybrid] F1 Score: {f1_score2:.3f}")
        print(f"[Strong Hybrid] Top-{top_n} Accuracy (HitRate@{top_n}): {topn_accuracy2:.3f} ({hits_topn2}/{evaluated_users2} users)")
        
        return {
            'precision_scores': precision_scores,
            'recall_scores': recall_scores,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'f1_score': f1_score,
            'topn_accuracy': topn_accuracy,
            'evaluated_users': evaluated_users,
            'top_n': top_n,
            'strong_hybrid': {
                'precision_scores': precision_scores2,
                'recall_scores': recall_scores2,
                'avg_precision': avg_precision2,
                'avg_recall': avg_recall2,
                'f1_score': f1_score2,
                'topn_accuracy': topn_accuracy2,
                'evaluated_users': evaluated_users2,
            }
        }

    def predict_item_properties_for_addtocart(self, user_id, n_predictions=5):
        """
        Core Task Implementation: Predict item properties for addtocart events using view events data.
        This addresses the main requirement from the task description.
        """
        if user_id not in self.user_item_matrix.index:
            return []
        
        # Get user's view events (what they've browsed)
        user_views = self.events_df[
            (self.events_df['visitorid'] == user_id) & 
            (self.events_df['event'] == 'view')
        ]['itemid'].tolist()
        
        if not user_views:
            return []
        
        # Get user's addtocart events (what they've actually wanted)
        user_addtocart = self.events_df[
            (self.events_df['visitorid'] == user_id) & 
            (self.events_df['event'] == 'addtocart')
        ]['itemid'].tolist()
        
        # Analyze user's viewing patterns to predict what they might add to cart
        if not self.item_properties_df.empty:
            # Get properties of items user has viewed
            viewed_item_props = self.item_properties_df[
                self.item_properties_df['itemid'].isin(user_views)
            ]
            
            if not viewed_item_props.empty:
                # Find most common properties in viewed items
                property_preferences = viewed_item_props.groupby(['property', 'value']).size().reset_index(name='count')
                property_preferences = property_preferences.sort_values('count', ascending=False)
                
                # Predict properties for addtocart based on viewing preferences
                predicted_properties = []
                for _, row in property_preferences.head(n_predictions).iterrows():
                    predicted_properties.append({
                        'property': row['property'],
                        'value': row['value'],
                        'confidence': row['count'] / len(user_views)  # Normalized confidence
                    })
                
                return predicted_properties
        
        return []
    
    def analyze_user_behavior_patterns(self):
        """
        Analyze user behavior patterns to understand viewing vs. addtocart behavior.
        This helps with the core prediction task.
        """
        print("\n=== User Behavior Pattern Analysis ===")
        
        # Analyze view to addtocart conversion patterns
        view_events = self.events_df[self.events_df['event'] == 'view']
        addtocart_events = self.events_df[self.events_df['event'] == 'addtocart']
        
        # Calculate conversion rates
        total_views = len(view_events)
        total_addtocart = len(addtocart_events)
        overall_conversion_rate = total_addtocart / total_views if total_views > 0 else 0
        
        print(f"Overall View to Addtocart Conversion Rate: {overall_conversion_rate:.4f}")
        print(f"Total Views: {total_views:,}")
        print(f"Total Add to Cart: {total_addtocart:,}")
        
        # Analyze by user segments
        user_conversion_rates = []
        for user_id in self.user_item_matrix.index:
            user_views = len(self.events_df[
                (self.events_df['visitorid'] == user_id) & 
                (self.events_df['event'] == 'view')
            ])
            user_addtocart = len(self.events_df[
                (self.events_df['visitorid'] == user_id) & 
                (self.events_df['event'] == 'addtocart')
            ])
            
            if user_views > 0:
                conversion_rate = user_addtocart / user_views
                user_conversion_rates.append({
                    'user_id': user_id,
                    'views': user_views,
                    'addtocart': user_addtocart,
                    'conversion_rate': conversion_rate
                })
        
        if user_conversion_rates:
            conversion_df = pd.DataFrame(user_conversion_rates)
            print(f"\nUser Conversion Rate Statistics:")
            print(f"Average conversion rate: {conversion_df['conversion_rate'].mean():.4f}")
            print(f"Median conversion rate: {conversion_df['conversion_rate'].median():.4f}")
            print(f"Users with high conversion (>0.1): {len(conversion_df[conversion_df['conversion_rate'] > 0.1])}")
            print(f"Users with low conversion (<0.01): {len(conversion_df[conversion_df['conversion_rate'] < 0.01])}")
        
        return {
            'overall_conversion_rate': overall_conversion_rate,
            'user_conversion_rates': user_conversion_rates if 'user_conversion_rates' in locals() else []
        }

    def evaluate_property_prediction(self, test_users=None, n_test_users=50):
        """
        Evaluate the property prediction model for addtocart events.
        This creates the evaluation metrics required by the task.
        """
        if test_users is None:
            # Select users with both view and addtocart events for evaluation
            users_with_both = []
            for user_id in self.user_item_matrix.index:
                views = len(self.events_df[
                    (self.events_df['visitorid'] == user_id) & 
                    (self.events_df['event'] == 'view')
                ])
                addtocart = len(self.events_df[
                    (self.events_df['visitorid'] == user_id) & 
                    (self.events_df['event'] == 'addtocart')
                ])
                if views > 0 and addtocart > 0:
                    users_with_both.append(user_id)
            
            test_users = users_with_both[:n_test_users]
        
        print(f"\n=== Property Prediction Evaluation ===")
        print(f"Evaluating property prediction for {len(test_users)} users")
        
        prediction_accuracy = []
        property_coverage = []
        
        for user_id in test_users:
            # Get actual addtocart items for this user
            actual_addtocart_items = self.events_df[
                (self.events_df['visitorid'] == user_id) & 
                (self.events_df['event'] == 'addtocart')
            ]['itemid'].tolist()
            
            if not actual_addtocart_items:
                continue
            
            # Get actual properties of addtocart items
            actual_properties = self.item_properties_df[
                self.item_properties_df['itemid'].isin(actual_addtocart_items)
            ]
            
            if actual_properties.empty:
                continue
            
            # Predict properties based on viewing history
            predicted_properties = self.predict_item_properties_for_addtocart(user_id, n_predictions=10)
            
            if predicted_properties:
                # Calculate prediction accuracy
                predicted_prop_values = set([(p['property'], p['value']) for p in predicted_properties])
                actual_prop_values = set([(row['property'], row['value']) for _, row in actual_properties.iterrows()])
                
                # Precision: how many predicted properties are actually in addtocart items
                if predicted_prop_values:
                    precision = len(predicted_prop_values & actual_prop_values) / len(predicted_prop_values)
                    prediction_accuracy.append(precision)
                
                # Coverage: how many actual properties were predicted
                if actual_prop_values:
                    coverage = len(predicted_prop_values & actual_prop_values) / len(actual_prop_values)
                    property_coverage.append(coverage)
        
        # Calculate overall metrics
        avg_accuracy = np.mean(prediction_accuracy) if prediction_accuracy else 0
        avg_coverage = np.mean(property_coverage) if property_coverage else 0
        
        print(f"Property Prediction Accuracy: {avg_accuracy:.3f}")
        print(f"Property Coverage: {avg_coverage:.3f}")
        print(f"Users evaluated: {len(prediction_accuracy)}")
        
        return {
            'prediction_accuracy': prediction_accuracy,
            'property_coverage': property_coverage,
            'avg_accuracy': avg_accuracy,
            'avg_coverage': avg_coverage
        }

def clean_and_save_datasets():
    """Data Cleaning + Save Cleaned Files"""
    print("\n" + "="*60)
    print("Data Cleaning + Save Cleaned Files")
    print("="*60)

    dtype_events = {
        "timestamp": str,
        "visitorid": str,
        "event": str,
        "itemid": str,
        "transactionid": str
    }
    dtype_cat = {"categoryid": str, "parentid": str}
    dtype_props = {"itemid": str, "property": str, "value": str}

    # -----------------------
    # Step 1: Load raw datasets
    # -----------------------
    events = pd.read_csv("events.csv", dtype=dtype_events, usecols=[c for c in EVENT_COLUMNS if c != 'transactionid' or 'transactionid' in pd.read_csv('events.csv', nrows=1).columns], nrows=MAX_ROWS_EVENTS)
    category_tree = pd.read_csv("category_tree.csv", dtype=dtype_cat)

    # Support multiple item properties file layouts
    if os.path.exists("item_properties.csv"):
        item_props = pd.read_csv("item_properties.csv", dtype=dtype_props, usecols=["itemid", "property", "value"], nrows=MAX_ROWS_ITEM_PROPERTIES)
    else:
        frames = []
        if os.path.exists("item_properties_part1.1.csv"):
            n1 = MAX_ROWS_ITEM_PROPERTIES // 2 if os.path.exists("item_properties_part2.csv") else MAX_ROWS_ITEM_PROPERTIES
            frames.append(pd.read_csv("item_properties_part1.1.csv", dtype=dtype_props, usecols=["itemid", "property", "value"], nrows=n1))
        if os.path.exists("item_properties_part2.csv"):
            n2 = MAX_ROWS_ITEM_PROPERTIES - (frames[0].shape[0] if frames else 0)
            if n2 > 0:
                frames.append(pd.read_csv("item_properties_part2.csv", dtype=dtype_props, usecols=["itemid", "property", "value"], nrows=n2))
        item_props = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(dtype_props.keys()))

    print("Initial shapes:")
    print("Events:", events.shape)
    print("Category Tree:", category_tree.shape)
    print("Item Properties:", item_props.shape)

    # -----------------------
    # Step 2: Clean abnormal values
    # -----------------------

    # Clean numeric-looking property values in item_properties
    def clean_numeric(val):
        if isinstance(val, str) and val.startswith("n"):
            try:
                return float(val[1:])
            except Exception:
                return np.nan
        return val

    if not item_props.empty:
        item_props["value"] = item_props["value"].apply(clean_numeric)
        item_props = item_props.dropna(subset=["itemid", "property", "value"])

    # -----------------------
    # Step 3: Outlier detection (abnormal users)
    # -----------------------
    def build_user_features(events_df):
        feats = []
        if events_df.empty:
            return pd.DataFrame(columns=["visitorid", "total_events", "views", "adds", "buys", "add_rate", "conv_rate"]).set_index("visitorid")
        for uid, g in events_df.groupby("visitorid"):
            total = len(g)
            views = (g["event"] == "view").sum()
            adds = (g["event"] == "addtocart").sum()
            buys = (g["event"] == "transaction").sum()
            feats.append({
                "visitorid": uid,
                "total_events": total,
                "views": views,
                "adds": adds,
                "buys": buys,
                "add_rate": adds/total if total > 0 else 0,
                "conv_rate": buys/total if total > 0 else 0
            })
        return pd.DataFrame(feats).set_index("visitorid")

    user_feats = build_user_features(events)

    if not user_feats.empty:
        iso = IsolationForest(contamination=0.02, random_state=42)
        user_feats["outlier"] = (iso.fit_predict(user_feats) == -1).astype(int)
        outlier_users = user_feats[user_feats["outlier"] == 1].index
        events = events[~events["visitorid"].isin(outlier_users)]
        print(f"Removed {len(outlier_users)} abnormal users.")
    else:
        print("No user features to build after cleaning. Skipping outlier detection.")

    # -----------------------
    # Step 4: Save cleaned datasets
    # -----------------------
    events.to_csv("events_cleaned.csv", index=False)
    category_tree.to_csv("category_tree_cleaned.csv", index=False)
    item_props.to_csv("item_properties_cleaned.csv", index=False)

    print("Cleaned datasets saved.")

def build_user_features_from_events(events_df, users_limit=None):
    """Build per-user feature matrix for anomaly detection from events DataFrame."""
    if events_df is None or events_df.empty:
        return pd.DataFrame()
    df = events_df
    if users_limit is not None and users_limit > 0:
        active_counts = df.groupby('visitorid').size().nlargest(users_limit)
        df = df[df['visitorid'].isin(active_counts.index)]
    grouped = df.groupby('visitorid')
    total_events = grouped.size().rename('total_events')
    views = grouped.apply(lambda g: (g['event'] == 'view').sum()).rename('views')
    adds = grouped.apply(lambda g: (g['event'] == 'addtocart').sum()).rename('adds')
    buys = grouped.apply(lambda g: (g['event'] == 'transaction').sum()).rename('buys')
    unique_items_viewed = grouped.apply(lambda g: g.loc[g['event'] == 'view', 'itemid'].nunique()).rename('unique_items_viewed')
    unique_items_bought = grouped.apply(lambda g: g.loc[g['event'] == 'transaction', 'itemid'].nunique()).rename('unique_items_bought')
    days_active = grouped.apply(lambda g: g['timestamp'].dt.date.nunique()).rename('days_active')
    features = pd.concat([
        total_events, views, adds, buys, unique_items_viewed, unique_items_bought, days_active
    ], axis=1).fillna(0)
    features['add_rate'] = features['adds'] / features['total_events'].replace(0, np.nan)
    features['buy_rate'] = features['buys'] / features['total_events'].replace(0, np.nan)
    features['view_to_add_rate'] = features['adds'] / features['views'].replace(0, np.nan)
    features['add_to_buy_rate'] = features['buys'] / features['adds'].replace(0, np.nan)
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)
    return features


def detect_abnormal_users_from_events(events_df, contamination=None, users_limit=None):
    """Detect abnormal users using IsolationForest; returns dict with features and flags."""
    if contamination is None:
        contamination = float(os.getenv('ANOMALY_CONTAMINATION', '0.02'))
    if users_limit is None:
        users_limit = int(os.getenv('ANOMALY_MAX_USERS', '50000'))
    features = build_user_features_from_events(events_df, users_limit=users_limit)
    if features.empty:
        return {
            'user_features': features,
            'labels': pd.Series(dtype=int),
            'scores': pd.Series(dtype=float),
            'flagged_user_ids': []
        }
    feature_columns = [
        'total_events','views','adds','buys',
        'unique_items_viewed','unique_items_bought','days_active',
        'add_rate','buy_rate','view_to_add_rate','add_to_buy_rate'
    ]
    X = features[feature_columns].astype(float).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = IsolationForest(contamination=contamination, random_state=42)
    labels = model.fit_predict(X_scaled)  # -1 outlier, 1 normal
    scores = -model.decision_function(X_scaled)  # higher = more anomalous
    features['outlier'] = (labels == -1).astype(int)
    features['anomaly_score'] = scores
    flagged_ids = features.index[features['outlier'] == 1].tolist()
    return {
        'user_features': features,
        'labels': pd.Series(labels, index=features.index),
        'scores': pd.Series(scores, index=features.index),
        'flagged_user_ids': flagged_ids
    }


def evaluate_anomaly_quality(events_df, user_features):
    """Evaluate anomaly detection quality using business-centric metrics and separation."""
    if user_features is None or user_features.empty:
        print("No user features available for evaluation.")
        return {}
    flagged_ids = set(user_features.index[user_features['outlier'] == 1])
    total_events = len(events_df)
    events_flagged = events_df[events_df['visitorid'].isin(flagged_ids)] if total_events else pd.DataFrame()
    events_normal = events_df[~events_df['visitorid'].isin(flagged_ids)] if total_events else pd.DataFrame()

    def conversion_rate(df):
        if df is None or df.empty:
            return 0.0
        buys = (df['event'] == 'transaction').sum()
        return buys / len(df)

    conv_flagged = float(conversion_rate(events_flagged))
    conv_normal = float(conversion_rate(events_normal))
    share_flagged_events = float(len(events_flagged) / total_events) if total_events else 0.0

    quality_score = 0.7 * max(0.0, (conv_normal - conv_flagged) / (conv_normal + 1e-9)) \
        + 0.3 * max(0.0, 1.0 - min(1.0, share_flagged_events / 0.25))

    silhouette = None
    try:
        num_flagged = int(user_features['outlier'].sum())
        num_normal = int(len(user_features) - num_flagged)
        if num_flagged > 1 and num_normal > 1:
            feat_cols = [
                'total_events','views','adds','buys',
                'unique_items_viewed','unique_items_bought','days_active',
                'add_rate','buy_rate','view_to_add_rate','add_to_buy_rate'
            ]
            X = StandardScaler().fit_transform(user_features[feat_cols].astype(float).values)
            y = user_features['outlier'].astype(int).values
            silhouette = float(silhouette_score(X, y))
    except Exception:
        silhouette = None

    results = {
        'num_flagged_users': int(user_features['outlier'].sum()),
        'share_flagged_users': float(user_features['outlier'].mean()) if len(user_features) else 0.0,
        'share_flagged_events': share_flagged_events,
        'conversion_rate_flagged': conv_flagged,
        'conversion_rate_normal': conv_normal,
        'quality_score': float(quality_score),
        'silhouette_separation': silhouette,
    }

    print("\n=== Anomaly Detection Evaluation ===")
    for k, v in results.items():
        print(f"{k}: {v}")
    return results

def one_hot_encode_categorical_ids(item_properties_df, id_suffix='id'):
    """
    One-hot encode all categorical features in item_properties_df whose property name ends with 'id'.
    Returns a DataFrame: index=itemid, columns=property_value, values=0/1.
    """
    if item_properties_df is None or item_properties_df.empty:
        return pd.DataFrame()
    # Find all property names ending with 'id' (e.g., categoryid, brandid)
    cat_properties = [p for p in item_properties_df['property'].unique() if p.lower().endswith(id_suffix)]
    if not cat_properties:
        print("No categorical ID properties found.")
        return pd.DataFrame()
    # Build one-hot matrix for each property
    onehot_frames = []
    for prop in cat_properties:
        prop_df = item_properties_df[item_properties_df['property'] == prop][['itemid', 'value']]
        # If items have multiple values for the same property, aggregate
        prop_df = prop_df.drop_duplicates()
        onehot = pd.get_dummies(prop_df.set_index('itemid')['value'], prefix=prop)
        onehot_frames.append(onehot)
    # Combine all one-hot matrices
    if onehot_frames:
        onehot_matrix = pd.concat(onehot_frames, axis=1).groupby(level=0).max().fillna(0).astype(int)
        return onehot_matrix
    else:
        return pd.DataFrame()

if __name__ == "__main__":
    print("This module defines RecommendationSystem. Run the UI with: streamlit run streamlit_app.py")
