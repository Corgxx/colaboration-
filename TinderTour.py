import streamlit as st
import requests
import json
import folium
import polyline
import random
from streamlit_folium import folium_static
from folium.plugins import MarkerCluster
import pandas as pd
from PIL import Image
import io
import base64
import os
from datetime import datetime
import numpy as np
from streamlit_card import card
from geopy.geocoders import Nominatim
import numpy as np
from sklearn.neighbors import NearestNeighbors
import pickle

# Set page configuration
st.set_page_config(
    page_title="Final Tinder - Activity Recommender",
    page_icon="🧭",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'current_index' not in st.session_state:
    st.session_state.current_index = 0
if 'liked_activities' not in st.session_state:
    st.session_state.liked_activities = []
if 'current_activities' not in st.session_state:
    st.session_state.current_activities = []
if 'location_coordinates' not in st.session_state:
    st.session_state.location_coordinates = None
if 'current_activity_type' not in st.session_state:
    st.session_state.current_activity_type = None
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {'liked_activities': [], 'preferences': {}}
if 'recommendation_model' not in st.session_state:
    st.session_state.recommendation_model = None

# API keys
YELP_API_KEY = "3CDquv2YIfVJBzNuoyau9lsDbqv21Zohj2ewjXvFlJDovEAGaiJWihHDxImRJatHXFA--wY1vVGuvLUX2bMyS-ipwsXjyDII3afybeUrgoisA1GbR8o0oOSB5bIYaHYx"  # Replace with your actual API key
OPENROUTE_API_KEY = "5b3ce3597851110001cf6248ee3c2977b26a41be8e066e26be3f95bf"  # Your OpenRouteService API key
def extract_features(activity):
    """Extract relevant features from an activity for ML processing"""
    features = []
    
    # Common features
    activity_type = activity.get('type', 'Business')
    
    if activity_type in ['Hiking', 'Cycling']:
        # Outdoor activity features
        features = [
            activity.get('distance_km', 0),
            activity.get('estimated_time_min', 0),
            # Convert difficulty to numeric value
            {'Easy': 1, 'Moderate': 2, 'Hard': 3}.get(activity.get('difficulty', 'Easy'), 1),
            activity.get('elevation_gain', 0)
        ]
    else:
        # Business activity features
        features = [
            activity.get('rating', 3.0),  # Default to 3.0 if not available
            len(activity.get('categories', [])),  # Number of categories
            {'$': 1, '$$': 2, '$$$': 3, '$$$$': 4}.get(activity.get('price', '$$'), 2),  # Price level
            activity.get('review_count', 0) / 100 if activity.get('review_count') else 0  # Normalized review count
        ]
        
    return np.array(features).reshape(1, -1)
# Function to save user preferences to user profile
def save_to_user_profile(activity):
    """Save liked activity and update user preferences"""
    # Add activity to liked activities list
    st.session_state.user_profile['liked_activities'].append({
        'id': activity.get('id', str(datetime.now())),
        'name': activity.get('name', 'Unnamed Activity'),
        'type': activity.get('type', 'Business'),
        'timestamp': datetime.now().isoformat(),
        'features': extract_features(activity).tolist()[0]
    })
    
    # Update preferences based on this activity
    activity_type = activity.get('type', 'Business')
    
    # Increment activity type counter
    if 'activity_types' not in st.session_state.user_profile['preferences']:
        st.session_state.user_profile['preferences']['activity_types'] = {}
    if activity_type not in st.session_state.user_profile['preferences']['activity_types']:
        st.session_state.user_profile['preferences']['activity_types'][activity_type] = 0
    st.session_state.user_profile['preferences']['activity_types'][activity_type] += 1
    
    # Track other preferences based on activity type
    if activity_type in ['Hiking', 'Cycling']:
        # Track preferred difficulty
        difficulty = activity.get('difficulty', 'Easy')
        if 'difficulty' not in st.session_state.user_profile['preferences']:
            st.session_state.user_profile['preferences']['difficulty'] = {}
        if difficulty not in st.session_state.user_profile['preferences']['difficulty']:
            st.session_state.user_profile['preferences']['difficulty'][difficulty] = 0
        st.session_state.user_profile['preferences']['difficulty'][difficulty] += 1
        
        # Track preferred distance range
        distance = activity.get('distance_km', 0)
        if 'distance_ranges' not in st.session_state.user_profile['preferences']:
            st.session_state.user_profile['preferences']['distance_ranges'] = {'short': 0, 'medium': 0, 'long': 0}
        if distance < 5:
            st.session_state.user_profile['preferences']['distance_ranges']['short'] += 1
        elif distance < 15:
            st.session_state.user_profile['preferences']['distance_ranges']['medium'] += 1
        else:
            st.session_state.user_profile['preferences']['distance_ranges']['long'] += 1
    else:
        # Track price preferences
        price = activity.get('price', '$$')
        if 'price' not in st.session_state.user_profile['preferences']:
            st.session_state.user_profile['preferences']['price'] = {}
        if price not in st.session_state.user_profile['preferences']['price']:
            st.session_state.user_profile['preferences']['price'][price] = 0
        st.session_state.user_profile['preferences']['price'][price] += 1
        
        # Track category preferences
        if 'categories' in activity:
            if 'categories' not in st.session_state.user_profile['preferences']:
                st.session_state.user_profile['preferences']['categories'] = {}
            for cat in activity['categories']:
                cat_title = cat['title']
                if cat_title not in st.session_state.user_profile['preferences']['categories']:
                    st.session_state.user_profile['preferences']['categories'][cat_title] = 0
                st.session_state.user_profile['preferences']['categories'][cat_title] += 1

# Function to train KNN model based on liked activities
def train_recommendation_model():
    """Train KNN model based on user's liked activities"""
    if len(st.session_state.user_profile['liked_activities']) < 3:
        # Need at least 3 liked activities to train a meaningful model
        return None
    
    # Separate activities by type since they have different feature sets
    outdoor_activities = []
    business_activities = []
    
    for activity in st.session_state.user_profile['liked_activities']:
        if activity['type'] in ['Hiking', 'Cycling']:
            outdoor_activities.append(activity)
        else:
            business_activities.append(activity)
    
    models = {}
    
    # Train outdoor activity model if enough data
    if len(outdoor_activities) >= 3:
        features = np.array([a['features'] for a in outdoor_activities])
        model = NearestNeighbors(n_neighbors=min(3, len(outdoor_activities)), algorithm='ball_tree')
        model.fit(features)
        models['outdoor'] = model
    
    # Train business activity model if enough data
    if len(business_activities) >= 3:
        features = np.array([a['features'] for a in business_activities])
        model = NearestNeighbors(n_neighbors=min(3, len(business_activities)), algorithm='ball_tree')
        model.fit(features)
        models['business'] = model
    
    return models if models else None

# Function to get personalized recommendations
def get_personalized_recommendations(activities):
    """Sort activities based on personalized recommendations"""
    if not st.session_state.recommendation_model or not activities:
        return activities
    
    # Determine activity type for first activity
    activity_type = activities[0].get('type', 'Business')
    model_type = 'outdoor' if activity_type in ['Hiking', 'Cycling'] else 'business'
    
    # If we don't have a model for this type, return original list
    if model_type not in st.session_state.recommendation_model:
        return activities
    
    # Extract features for all activities
    features_list = []
    for activity in activities:
        features_list.append(extract_features(activity)[0])
    
    features_array = np.array(features_list)
    
    # Get model for this activity type
    model = st.session_state.recommendation_model[model_type]
    
    # Calculate distances to liked activities
    distances, _ = model.kneighbors(features_array)
    
    # Calculate average distance for each activity (smaller is better)
    avg_distances = np.mean(distances, axis=1)
    
    # Sort activities by similarity (smaller distance is better)
    sorted_indices = np.argsort(avg_distances)
    sorted_activities = [activities[i] for i in sorted_indices]
    
    return sorted_activities
# Function to get coordinates from location name
def get_coordinates(location_name):
    try:
        geolocator = Nominatim(user_agent="final_tinder_app")
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
        else:
            st.error(f"Could not find coordinates for {location_name}")
            return None
    except Exception as e:
        st.error(f"Error getting coordinates: {e}")
        return None

# Function to fetch activities from Yelp API
def fetch_yelp_activities(latitude, longitude, category, radius=10000, limit=20):
    # Original code to fetch activities from Yelp API
    url = "https://api.yelp.com/v3/businesses/search"
    
    headers = {
        "Authorization": f"Bearer {YELP_API_KEY}"
    }
    
    category_map = {
        "Restaurant": "restaurants",
        "Coffee & Drinks": "coffee,cafes",
        "Bar": "bars,pubs",
        "Hotel / Stay": "hotels"
    }
    
    category_alias = category_map.get(category, category)
    
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "categories": category_alias,
        "radius": radius,
        "limit": limit,
        "sort_by": "rating"
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        if response.status_code == 200:
            activities = response.json().get("businesses", [])
            
            # Apply personalized recommendations if model exists
            if st.session_state.recommendation_model:
                activities = get_personalized_recommendations(activities)
            
            return activities
        else:
            st.error(f"Error fetching from Yelp API: {response.status_code}, {response.text}")
            return []
    except Exception as e:
        st.error(f"Exception when calling Yelp API: {e}")
        return []
# Function to fetch hiking trails
def fetch_hiking_routes(latitude, longitude, radius=10000):
    # Original code to fetch/generate hiking routes
    sample_routes = []
    
    # Create 5 sample routes
    for i in range(5):
        # Generate start point with slight offset from provided coordinates
        start_lat = latitude + (random.random() - 0.5) * 0.05
        start_lon = longitude + (random.random() - 0.5) * 0.05
        
        # Generate end point
        end_lat = start_lat + (random.random() - 0.5) * 0.02
        end_lon = start_lon + (random.random() - 0.5) * 0.02
        
        # Calculate the coordinates for the route (simplified as a straight line for demo)
        num_points = 10
        route_coordinates = []
        for j in range(num_points):
            point_lat = start_lat + (end_lat - start_lat) * j / (num_points - 1)
            point_lon = start_lon + (end_lon - start_lon) * j / (num_points - 1)
            route_coordinates.append([point_lat, point_lon])
        
        # Calculate estimated distance and time
        distance_km = round(random.uniform(2, 15), 1)  # Random distance between 2 and 15 km
        estimated_time = round(distance_km / 4 * 60)  # Assuming 4 km/h average speed
        
        # Create route object
        route = {
            "id": f"hiking_{i}",
            "name": f"Hiking Trail {i+1}",
            "coordinates": route_coordinates,
            "distance_km": distance_km,
            "estimated_time_min": estimated_time,
            "difficulty": random.choice(["Easy", "Moderate", "Hard"]),
            "elevation_gain": round(random.uniform(50, 500)),  # Random elevation gain in meters
            "type": "Hiking",
            "image_url": None  # We'll generate this from the map
        }
        
        sample_routes.append(route)
    
    # Apply personalized recommendations if model exists
    if st.session_state.recommendation_model:
        sample_routes = get_personalized_recommendations(sample_routes)
    
    return sample_routes

# Function to fetch cycling routes
def fetch_cycling_routes(latitude, longitude, radius=10000):
    # Similar to hiking, but with longer routes and faster expected speeds
    sample_routes = []
    
    for i in range(5):
        start_lat = latitude + (random.random() - 0.5) * 0.05
        start_lon = longitude + (random.random() - 0.5) * 0.05
        
        end_lat = start_lat + (random.random() - 0.5) * 0.04
        end_lon = start_lon + (random.random() - 0.5) * 0.04
        
        num_points = 15
        route_coordinates = []
        for j in range(num_points):
            point_lat = start_lat + (end_lat - start_lat) * j / (num_points - 1)
            point_lon = start_lon + (end_lon - start_lon) * j / (num_points - 1)
            route_coordinates.append([point_lat, point_lon])
        
        distance_km = round(random.uniform(5, 30), 1)  # Cycling routes are typically longer
        estimated_time = round(distance_km / 15 * 60)  # Assuming 15 km/h average speed
        
        route = {
            "id": f"cycling_{i}",
            "name": f"Cycling Route {i+1}",
            "coordinates": route_coordinates,
            "distance_km": distance_km,
            "estimated_time_min": estimated_time,
            "difficulty": random.choice(["Easy", "Moderate", "Hard"]),
            "elevation_gain": round(random.uniform(100, 800)),
            "type": "Cycling",
            "image_url": None
        }
        
        sample_routes.append(route)
    
    return sample_routes

# Function to get actual routes from OpenRouteService API
def get_openroute_service_route(start_coords, end_coords, profile='foot-hiking'):
    """
    Get a route from OpenRouteService API
    profiles: foot-hiking, cycling-regular, cycling-mountain, cycling-road
    """
    base_url = "https://api.openrouteservice.org/v2/directions/"
    
    # Reorder coordinates for the API [longitude, latitude]
    start = f"{start_coords[1]},{start_coords[0]}"
    end = f"{end_coords[1]},{end_coords[0]}"
    
    url = f"{base_url}{profile}"
    headers = {
        'Authorization': OPENROUTE_API_KEY,
        'Content-Type': 'application/json'
    }
    
    body = {
        "coordinates": [[float(start_coords[1]), float(start_coords[0])], 
                        [float(end_coords[1]), float(end_coords[0])]],
        "elevation": "true"
    }
    
    try:
        response = requests.post(url, json=body, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            if 'routes' in data and len(data['routes']) > 0:
                route = data['routes'][0]
                geometry = route['geometry']
                # Convert encoded polyline to coordinates
                coords = polyline.decode(geometry, geojson=True)
                
                # Extract distance and duration
                distance_km = route['summary']['distance'] / 1000
                duration_min = route['summary']['duration'] / 60
                
                # Extract elevation data if available
                elevation_gain = 0
                if 'ascent' in route['summary']:
                    elevation_gain = route['summary']['ascent']
                
                return {
                    'coordinates': coords,
                    'distance_km': distance_km,
                    'duration_min': duration_min,
                    'elevation_gain': elevation_gain
                }
            else:
                st.error("No routes found in the response")
                return None
        else:
            st.error(f"Error from OpenRouteService API: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        st.error(f"Exception when calling OpenRouteService API: {e}")
        return None

# Function to create a map for a route
def create_route_map(coordinates, start_icon='play', end_icon='stop'):
    # Create map centered at the middle point of the route
    mid_point = coordinates[len(coordinates)//2]
    m = folium.Map(location=mid_point, zoom_start=13)
    
    # Add the route line
    folium.PolyLine(
        coordinates,
        color='blue',
        weight=5,
        opacity=0.7
    ).add_to(m)
    
    # Add markers for start and end points
    start_point = coordinates[0]
    end_point = coordinates[-1]
    
    folium.Marker(
        location=start_point,
        icon=folium.Icon(color='green', icon=start_icon),
        popup='Start'
    ).add_to(m)
    
    folium.Marker(
        location=end_point,
        icon=folium.Icon(color='red', icon=end_icon),
        popup='End'
    ).add_to(m)
    
    return m
def add_user_profile_ui():
    with st.sidebar:
        st.divider()
        st.subheader("📊 User Profile")
        
        # Show basic stats
        total_liked = len(st.session_state.user_profile['liked_activities'])
        st.write(f"Total liked activities: {total_liked}")
        
        if total_liked > 0:
            # Show top preferences if available
            with st.expander("Your Preferences"):
                prefs = st.session_state.user_profile['preferences']
                
                # Show preferred activity types
                if 'activity_types' in prefs and prefs['activity_types']:
                    st.write("**Favorite activity types:**")
                    sorted_types = sorted(prefs['activity_types'].items(), key=lambda x: x[1], reverse=True)
                    for t, count in sorted_types[:3]:
                        st.write(f"- {t}: {count} likes")
                
                # Show other preferences based on what's available
                if 'difficulty' in prefs and prefs['difficulty']:
                    st.write("**Preferred difficulty:**")
                    sorted_diff = sorted(prefs['difficulty'].items(), key=lambda x: x[1], reverse=True)
                    st.write(f"- {sorted_diff[0][0]}")
                
                if 'price' in prefs and prefs['price']:
                    st.write("**Price preference:**")
                    sorted_price = sorted(prefs['price'].items(), key=lambda x: x[1], reverse=True)
                    st.write(f"- {sorted_price[0][0]}")
        
        # Export/Import buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Profile"):
                if export_user_profile():
                    st.success("Profile saved!")
        
        with col2:
            if st.button("📂 Load Profile"):
                if import_user_profile():
                    st.success("Profile loaded!")
                else:
                    st.warning("No profile found.")
# Function to convert Folium map to image
def map_to_image(m):
    # This is a placeholder. In a real app, you would need to render the map
    # and convert it to an image. For simplicity, we'll use a placeholder.
    return "https://via.placeholder.com/400x300?text=Map+Preview"

# Function to like an activity
def like_activity():
    if st.session_state.current_activities and st.session_state.current_index < len(st.session_state.current_activities):
        current_activity = st.session_state.current_activities[st.session_state.current_index]
        
        # Add to liked activities list (original functionality)
        st.session_state.liked_activities.append(current_activity)
        
        # Save to user profile and update preferences
        save_to_user_profile(current_activity)
        
        # Train recommendation model if enough data
        st.session_state.recommendation_model = train_recommendation_model()
        
        # Move to next activity
        st.session_state.current_index += 1
        st.session_state.image_index = 0  # Reset image index for next activity

# Function to export user profile to file
def export_user_profile():
    """Export user profile to a file"""
    try:
        with open('user_profile.pkl', 'wb') as f:
            pickle.dump(st.session_state.user_profile, f)
        return True
    except Exception as e:
        st.error(f"Error saving user profile: {e}")
        return False

# Function to import user profile from file
def import_user_profile():
    """Import user profile from a file"""
    try:
        if os.path.exists('user_profile.pkl'):
            with open('user_profile.pkl', 'rb') as f:
                st.session_state.user_profile = pickle.load(f)
            
            # Retrain model with loaded data
            st.session_state.recommendation_model = train_recommendation_model()
            return True
        return False
    except Exception as e:
        st.error(f"Error loading user profile: {e}")
        return False

# Function to pass on an activity
def pass_activity():
    if st.session_state.current_activities and st.session_state.current_index < len(st.session_state.current_activities):
        st.session_state.current_index += 1
        st.session_state.image_index = 0  # Reset image index for next activity

# Function to restart the app
def restart_app():
    st.session_state.current_index = 0
    st.session_state.current_activities = []
    st.session_state.image_index = 0

# Function to display current activity card
def display_activity_card():
    if not st.session_state.current_activities or st.session_state.current_index >= len(st.session_state.current_activities):
        st.info("No more activities to show. Try changing your filters or location!")
        return
    if st.session_state.recommendation_model:
        st.markdown("✨ **Recommendations personalized based on your likes**")
    
    activity = st.session_state.current_activities[st.session_state.current_index]
    
    # Container for the card
    with st.container():
        # Display the image carousel or map
        col1, col2, col3 = st.columns([1, 4, 1])
        
        with col1:
            # Previous image button
            if st.button("◀️", key="prev_img"):
                if 'photos' in activity and len(activity['photos']) > 1:
                    st.session_state.image_index = (st.session_state.image_index - 1) % len(activity['photos'])
        
        with col2:
            activity_type = activity.get('type', 'Business')
            
            if activity_type in ['Hiking', 'Cycling']:
                # Display route map
                if 'coordinates' in activity:
                    m = create_route_map(activity['coordinates'])
                    folium_static(m, width=600, height=400)
            else:
                # Display business image
                image_url = activity.get('image_url')
                if image_url:
                    st.image(image_url, use_column_width=True)
                else:
                    st.image("https://via.placeholder.com/600x400?text=No+Image+Available", use_column_width=True)
        
        with col3:
            # Next image button
            if st.button("▶️", key="next_img"):
                if 'photos' in activity and len(activity['photos']) > 1:
                    st.session_state.image_index = (st.session_state.image_index + 1) % len(activity['photos'])
        
        # Activity info box
        st.subheader(activity.get('name', 'Unnamed Activity'))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if activity_type in ['Hiking', 'Cycling']:
                st.write(f"**Type:** {activity_type}")
                st.write(f"**Distance:** {activity.get('distance_km', 0):.1f} km")
                st.write(f"**Est. Time:** {activity.get('estimated_time_min', 0):.0f} min")
                st.write(f"**Difficulty:** {activity.get('difficulty', 'N/A')}")
                st.write(f"**Elevation Gain:** {activity.get('elevation_gain', 0)} m")
            else:
                if 'location' in activity and 'display_address' in activity['location']:
                    address = ", ".join(activity['location']['display_address'])
                    st.write(f"**Address:** {address}")
                
                rating = activity.get('rating', 'N/A')
                if rating != 'N/A':
                    stars = "⭐" * int(rating) + ("½" if rating % 1 >= 0.5 else "")
                    st.write(f"**Rating:** {rating} {stars}")
                
                price = activity.get('price', 'N/A')
                st.write(f"**Price:** {price}")
                
                if 'distance' in activity:
                    distance_km = activity['distance'] / 1000
                    st.write(f"**Distance:** {distance_km:.1f} km")
        
        with col2:
            if activity_type not in ['Hiking', 'Cycling']:
                if 'categories' in activity:
                    categories = ", ".join([cat['title'] for cat in activity['categories']])
                    st.write(f"**Categories:** {categories}")
                
                if 'display_phone' in activity:
                    st.write(f"**Phone:** {activity['display_phone']}")
                
                if 'url' in activity:
                    st.write(f"[View on Yelp]({activity['url']})")
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("👎 Pass", key="pass_btn", use_container_width=True):
                pass_activity()
                st.rerun()
        
        with col3:
            if st.button("👍 Like", key="like_btn", use_container_width=True):
                like_activity()
                st.rerun()

# Function to display liked activities
def display_liked_activities():
    if not st.session_state.liked_activities:
        st.info("You haven't liked any activities yet. Start swiping!")
        return
    
    st.subheader("Your Liked Activities")
    
    for i, activity in enumerate(st.session_state.liked_activities):
        with st.expander(f"{i+1}. {activity.get('name', 'Unnamed Activity')}"):
            activity_type = activity.get('type', 'Business')
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if activity_type in ['Hiking', 'Cycling']:
                    # Display route map
                    if 'coordinates' in activity:
                        m = create_route_map(activity['coordinates'])
                        folium_static(m, width=300, height=200)
                else:
                    # Display business image
                    image_url = activity.get('image_url')
                    if image_url:
                        st.image(image_url, width=300)
                    else:
                        st.image("https://via.placeholder.com/300x200?text=No+Image+Available", width=300)
            
            with col2:
                if activity_type in ['Hiking', 'Cycling']:
                    st.write(f"**Type:** {activity_type}")
                    st.write(f"**Distance:** {activity.get('distance_km', 0):.1f} km")
                    st.write(f"**Est. Time:** {activity.get('estimated_time_min', 0):.0f} min")
                    st.write(f"**Difficulty:** {activity.get('difficulty', 'N/A')}")
                    st.write(f"**Elevation Gain:** {activity.get('elevation_gain', 0)} m")
                else:
                    if 'location' in activity and 'display_address' in activity['location']:
                        address = ", ".join(activity['location']['display_address'])
                        st.write(f"**Address:** {address}")
                    
                    rating = activity.get('rating', 'N/A')
                    if rating != 'N/A':
                        stars = "⭐" * int(rating) + ("½" if rating % 1 >= 0.5 else "")
                        st.write(f"**Rating:** {rating} {stars}")
                    
                    price = activity.get('price', 'N/A')
                    st.write(f"**Price:** {price}")
                    
                    if 'categories' in activity:
                        categories = ", ".join([cat['title'] for cat in activity['categories']])
                        st.write(f"**Categories:** {categories}")
                    
                    if 'display_phone' in activity:
                        st.write(f"**Phone:** {activity['display_phone']}")
                    
                    if 'url' in activity:
                        st.write(f"[View on Yelp]({activity['url']})")

# Main application
def main():
    # App header
    st.title("🧭 Final Tinder - Activity Recommender")
    st.write("Swipe through activities like you're finding your next date!")
    
    # Sidebar for filters and options
    with st.sidebar:
        st.header("Find Your Next Activity")
        
        # Location input
        location_input = st.text_input("Enter Location (City, Address, etc.)", "New York")
        
        # Get coordinates button
        if st.button("Set Location"):
            coordinates = get_coordinates(location_input)
            if coordinates:
                st.session_state.location_coordinates = coordinates
                st.success(f"Location set to: {location_input} ({coordinates[0]:.4f}, {coordinates[1]:.4f})")
                # Reset current index and activities
                st.session_state.current_index = 0
                st.session_state.current_activities = []
        
        # Distance range slider
        radius = st.slider("Distance Range (km)", min_value=1, max_value=40, value=10)
        radius_meters = radius * 1000  # Convert to meters for API
        
        # Activity type selector
        activity_type = st.selectbox(
            "Activity Type",
            ["Restaurant", "Coffee & Drinks", "Bar", "Hotel / Stay", "Hiking", "Cycling"]
        )
        
        # Button to fetch activities
        if st.button("Find Activities"):
            if st.session_state.location_coordinates:
                latitude, longitude = st.session_state.location_coordinates
                
                with st.spinner("Fetching activities..."):
                    if activity_type in ["Restaurant", "Coffee & Drinks", "Bar", "Hotel / Stay"]:
                        activities = fetch_yelp_activities(latitude, longitude, activity_type, radius_meters)
                    elif activity_type == "Hiking":
                        activities = fetch_hiking_routes(latitude, longitude, radius_meters)
                    elif activity_type == "Cycling":
                        activities = fetch_cycling_routes(latitude, longitude, radius_meters)
                    
                    if activities:
                        st.session_state.current_activities = activities
                        st.session_state.current_index = 0
                        st.session_state.current_activity_type = activity_type
                        st.success(f"Found {len(activities)} {activity_type} activities!")
                    else:
                        st.error(f"No {activity_type} activities found. Try expanding your search radius or changing location.")
            else:
                st.error("Please set a location first!")
        
        # Show liked activities button
        if st.button("View Liked Activities"):
            st.session_state.show_liked = True
        else:
            st.session_state.show_liked = False
            
        # Reset button
        if st.button("Reset All"):
            st.session_state.clear()
            st.experimental_rerun()
    add_user_profile_ui()

    # Main content area
    if 'show_liked' in st.session_state and st.session_state.show_liked:
        display_liked_activities()
    else:
        # Initial prompt if no location is set
        if not st.session_state.location_coordinates:
            st.info("👈 Start by setting your location in the sidebar!")
        # Display activities if available
        elif st.session_state.current_activities:
            display_activity_card()
        else:
            st.info("👈 Select an activity type and click 'Find Activities' to start browsing!")

if __name__ == "__main__":
    main()