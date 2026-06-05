import requests

GROQ_API_KEY = "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"

def llm_check(news):

    prompt = f"""
Determine if the following news is FAKE or REAL.

News:
{news}

Answer REAL or FAKE and explain briefly.
"""

    try:

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        )

        result = response.json()

        print("Groq API Response:", result)   # 👈 THIS WILL SHOW THE ERROR

        if "choices" in result:
            return result["choices"][0]["message"]["content"]

        if "error" in result:
            return result["error"]["message"]

        return "Unknown LLM error"

    except Exception as e:
        return str(e)