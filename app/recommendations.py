import pandas as pd
import streamlit as st
from groq import Groq

def generate_safety_recommendations(braking_data: pd.DataFrame, swerving_data: pd.DataFrame) -> pd.DataFrame:
    try:
        client = Groq(api_key=st.secrets["GROQ_API_KEY"])
        braking_hotspots = braking_data.nlargest(5, "intensity")[["hotspot_id", "lat", "lon", "intensity", "incidents_count", "avg_deceleration", "road_type", "surface_quality"]]
        swerving_hotspots = swerving_data.nlargest(5, "intensity")[["hotspot_id", "lat", "lon", "intensity", "incidents_count", "avg_lateral_movement", "road_type", "obstruction_present"]]
        prompt = (
            "Generate safety recommendations for cycling based on the following top 5 braking and swerving hotspots data. "
            "Each recommendation should include fields: Location, Issue, Recommendation, Priority, Type, ROI_estimate. "
            "Consider road_type, surface_quality, obstruction_present, intensity, incidents_count, avg_deceleration, and avg_lateral_movement. "
            "Output as a JSON array.\n"
            f"Braking Hotspots:\n{braking_hotspots.to_string()}\n"
            f"Swerving Hotspots:\n{swerving_hotspots.to_string()}"
        )
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama3-8b-8192",
            max_tokens=2000,
            temperature=0.7
        )
        response = chat_completion.choices[0].message.content
        try:
            recommendations = pd.read_json(response, orient="records")
            required_columns = ['Location', 'Issue', 'Recommendation', 'Priority', 'Type', 'ROI_estimate']
            if not all(col in recommendations.columns for col in required_columns):
                raise ValueError("Generated recommendations missing required columns")
            return recommendations
        except ValueError as e:
            st.error(f"Error parsing recommendations: {str(e)}")
            return pd.DataFrame(columns=['Location', 'Issue', 'Recommendation', 'Priority', 'Type', 'ROI_estimate'])
    except Exception as e:
        st.error(f"Error generating recommendations: {str(e)}. Check GROQ_API_KEY or network connection.")
        return pd.DataFrame(columns=['Location', 'Issue', 'Recommendation', 'Priority', 'Type', 'ROI_estimate'])
