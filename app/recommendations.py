import pandas as pd
import streamlit as st
from groq import Groq
import os
import json
import re

def generate_safety_recommendations(braking_data: pd.DataFrame, swerving_data: pd.DataFrame) -> pd.DataFrame:
    """
    Generate safety recommendations using Groq API based on braking and swerving hotspot data.
    
    Args:
        braking_data (pd.DataFrame): Braking hotspot data with columns: hotspot_id, lat, lon, intensity, incidents_count, avg_deceleration, road_type, surface_quality, date_recorded
        swerving_data (pd.DataFrame): Swerving hotspot data with columns: hotspot_id, lat, lon, intensity, incidents_count, avg_lateral_movement, road_type, obstruction_present, date_recorded
    
    Returns:
        pd.DataFrame: Recommendations with columns: Location, Issue, Recommendation, Priority, Type, ROI_estimate
    """
    required_columns = ['Location', 'Issue', 'Recommendation', 'Priority', 'Type', 'ROI_estimate']
    empty_df = pd.DataFrame(columns=required_columns)
    
    try:
        # Initialize Groq client
        api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
        if not api_key:
            st.error("GROQ_API_KEY is not set in Streamlit secrets or environment variables.")
            return empty_df
        
        client = Groq(api_key=api_key)
        
        # Select top 5 hotspots for braking and swerving
        braking_hotspots = braking_data.nlargest(5, "intensity")[["hotspot_id", "lat", "lon", "intensity", "incidents_count", "avg_deceleration", "road_type", "surface_quality"]]
        swerving_hotspots = swerving_data.nlargest(5, "intensity")[["hotspot_id", "lat", "lon", "intensity", "incidents_count", "avg_lateral_movement", "road_type", "obstruction_present"]]
        
        # Debug: Display input data
        if st.session_state.get("debug_mode", False):
            st.write("Braking Hotspots:", braking_hotspots)
            st.write("Swerving Hotspots:", swerving_hotspots)
        
        # Prepare prompt with strict JSON format instruction
        prompt = (
            "You are a cycling safety expert. Generate safety recommendations for cycling based on the following top 5 braking and swerving hotspots data. "
            "Each recommendation must include fields: Location, Issue, Recommendation, Priority (High/Medium/Low), Type (Infrastructure/Maintenance/Policy), ROI_estimate (High/Medium/Low). "
            "Consider road_type, surface_quality, obstruction_present, intensity, incidents_count, avg_deceleration, and avg_lateral_movement when formulating recommendations. "
            "Return the response as a JSON array of objects, e.g., [{'Location': '...', 'Issue': '...', ...}, ...]. "
            "Do NOT wrap the JSON in markdown code blocks (e.g., ```json) or include any additional text outside the JSON array. "
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
        
        # Get response
        response = chat_completion.choices[0].message.content
        
        # Debug: Display raw response
        if st.session_state.get("debug_mode", False):
            st.write("Raw Groq API Response:", response)
        
        # Clean response: Remove markdown code blocks or leading/trailing text
        response = re.sub(r'^```json\s*|\s*```$', '', response.strip())
        
        # Parse JSON
        try:
            recommendations_data = json.loads(response)
            if not isinstance(recommendations_data, list):
                st.error("Groq API response is not a JSON array. Expected a list of recommendation objects.")
                return empty_df
            
            recommendations = pd.DataFrame(recommendations_data)
            if not all(col in recommendations.columns for col in required_columns):
                st.error(f"Generated recommendations missing required columns: {', '.join(set(required_columns) - set(recommendations.columns))}")
                return empty_df
            
            return recommendations
        except json.JSONDecodeError as e:
            st.error(f"Error parsing Groq API response as JSON: {str(e)}. Response may be malformed.")
            if st.session_state.get("debug_mode", False):
                st.write("Malformed Response:", response)
            return empty_df
    
    except Exception as e:
        # Handle specific errors
        if "proxies" in str(e).lower():
            st.error("Groq API error: Incompatible 'httpx' version. Ensure 'httpx==0.27.2' is installed.")
        elif "401" in str(e).lower():
            st.error("Groq API error: Invalid GROQ_API_KEY. Please verify the API key in Streamlit secrets.")
        else:
            st.error(f"Error generating recommendations: {str(e)}. Check GROQ_API_KEY or network connection.")
        return empty_df
