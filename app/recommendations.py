import requests
import pandas as pd
import os
import json

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_API_URL = "https://api.groq.com/v1/chat/completions"  # Replace with actual endpoint if different

def generate_safety_recommendations(braking_data, swerving_data, model="llama-3-70b-8192"):
    """
    Generate safety recommendations by sending hotspot data to the Groq API.
    """
    # Prepare a concise prompt with top hotspots
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
        "Authorization": f"Bearer {GROQ_API_KEY}",
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

    response = requests.post(GROQ_API_URL, headers=headers, json=body)
    response.raise_for_status()
    response_data = response.json()

    # Extract and parse the AI's recommendations
    try:
        llm_reply = response_data['choices'][0]['message']['content']
        recommendations = json.loads(llm_reply)
        return pd.DataFrame(recommendations)
    except Exception as e:
        print("Error processing Groq response:", e)
        return pd.DataFrame([{"location": "Error", "issue": str(e), "recommendation": "See logs.", "priority": "N/A", "type": "N/A", "ROI_estimate": "N/A"}])
