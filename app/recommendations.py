import streamlit as st
import requests
import pandas as pd
import json

GROQ_API_URL = "https://api.groq.com/v1/chat/completions"  # Update if Groq's endpoint changes

def generate_safety_recommendations(braking_data, swerving_data, model="llama-3-70b-8192"):
    """
    Generate safety recommendations by sending hotspot data to Groq API (using key from Streamlit secrets).
    Args:
        braking_data (pd.DataFrame): Braking hotspot data.
        swerving_data (pd.DataFrame): Swerving hotspot data.
        model (str): Groq model name.
    Returns:
        pd.DataFrame: LLM-generated recommendations.
    """
    # Get API key securely from Streamlit secrets
    groq_api_key = st.secrets["GROQ_API_KEY"]

    # Prepare the prompt
    braking_top = braking_data.sort_values('intensity', ascending=False).head(5).to_dict(orient='records')
    swerving_top = swerving_data.sort_values('intensity', ascending=False).head(5).to_dict(orient='records')

    prompt = (
        "You are a cycling infrastructure safety expert. "
        "Given the following braking and swerving hotspot data, provide actionable and concise safety recommendations for each hotspot. "
        "Return the results as a JSON list with fields: location, issue, recommendation, priority, type, ROI_estimate.\n\n"
        f"Braking Hotspots:\n{json.dumps(braking_top, indent=2)}\n\n"
        f"Swerving Hotspots:\n{json.dumps(swerving_top, indent=2)}"
    )

    headers = {
        "Authorization": f"Bearer {groq_api_key}",
        "Content-Type": "application/json"
    }

    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You generate urban cycling safety recommendations."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024,
        "temperature": 0.4
    }

    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=body, timeout=60)
        response.raise_for_status()
        response_data = response.json()

        # Extract and parse the AI's recommendations
        llm_reply = response_data['choices'][0]['message']['content']
        recommendations = json.loads(llm_reply)
        return pd.DataFrame(recommendations)
    except Exception as e:
        st.error(f"Error getting recommendations from Groq API: {e}")
        return pd.DataFrame([{
            "location": "Error",
            "issue": str(e),
            "recommendation": "See logs.",
            "priority": "N/A",
            "type": "N/A",
            "ROI_estimate": "N/A"
        }])
