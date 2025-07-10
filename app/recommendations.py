import pandas as pd
import streamlit as st
from groq import Groq
import os

def generate_safety_recommendations(braking_data: pd.DataFrame, swerving_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate safety recommendations using Groq API based on braking and swerving hotspot data.
    
    Args:
        braking_data (pd.DataFrame): Braking hotspot data with columns: hotspot_id, lat, lon, intensity, incidents_count, avg_deceleration, road_type, surface_quality, date_recorded
        swerving_data (pd.DataFrame): Swerving hotspot data with columns: hotspot_id, lat, lon, intensity, incidents_count, avg_lateral_movement, road_type, obstruction_present, date_recorded
    
    Returns:
        pd.DataFrame: Recommendations with columns: Location, Issue, Recommendation, Priority, Type, ROI_estimate
    """
    try:
        # Initialize Groq client
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY is not set in Streamlit secrets or environment variables.")
            return pd.DataFrame(columns=['Location', 'Issue', 'Recommendation', 'Priority', 'Type', 'ROI_estimate'])
        
        client = Groq(api_key=api_key)
        
        # Select top 5 hotspots for braking and swerving
        braking_hotspots = braking_data.nlargest(5, "intensity")[["hotspot_id", "lat", "lon", "intensity", "incidents_count", "avg_deceleration", "road_type", "surface_quality"]]
        swerving_hotspots = swerving_data.nlargest(5, "intensity")[["hotspot_id", "lat", "lon", "intensity", "incidents_count", "avg_lateral_movement", "road_type", "obstruction_present"]]
        
        # Prepare prompt for Groq API
        prompt = (
            "You are a cycling safety expert. Generate safety recommendations for cycling based on the following top 5 braking and swerving hotspots data. "
            "Each recommendation should include fields: Location, Issue, Recommendation, Priority (High/Medium/Low), Type (Infrastructure/Maintenance/Policy), ROI_estimate (High/Medium/Low). "
            "Consider road_type, surface_quality, obstruction_present, intensity, incidents_count, avg_deceleration, and avg_lateral_movement when formulating recommendations. "
            "Output as a JSON array with clear, actionable recommendations tailored to the data.\n"
            f"Braking Hotspots:\n{braking_hotspots.to_string()}\n"
            f"Swerving Hotspots:\n{swerving_hotspots.to_string()}"
        )
        
        # Make API call
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            max_tokens=2000,
            temperature=0.7
        )
        
        # Parse response
        response = chat_completion.choices[0].message.content
        try:
            recommendations = pd.read_json(response, orient="records")
            required_columns = ['Location', 'Issue', 'Recommendation', 'Priority', 'Type', 'ROI_estimate']
            if not all(col in recommendations.columns for col in required_columns):
                st.error(f"Generated recommendations missing required columns: {', '.join(set(required_columns) - set(recommendations.columns))}")
                return pd.DataFrame(columns=required_columns)
            return recommendations
        except ValueError as e:
            st.error(f"Error parsing Groq API response as JSON: {str(e)}")
            return pd.DataFrame(columns=['Location', 'Issue', 'Recommendation', 'Priority', 'Type', 'ROI_estimate'])
    
    except Exception as e:
        # Specific handling for proxies error
        if "proxies" in str(e).lower():
            st.error("Groq API error: Incompatible 'httpx' version. Ensure 'httpx==0.27.2' is installed in your environment.")
        else:
            st.error(f"Error generating recommendations: {str(e)}. Check GROQ_API_KEY or network connection.")
        return pd.DataFrame(columns=['Location', 'Issue', 'Recommendation', 'Priority', 'Type', 'ROI_estimate'])
