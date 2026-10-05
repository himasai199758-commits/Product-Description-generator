# Product Description Generator

A Streamlit app that generates product descriptions with Google's Gemini API, supports one product or CSV batches, optionally analyses public product image URLs, and exports results to CSV.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and this README.
2. In Streamlit Community Cloud, choose **Create app**, select your repository and branch, and set the main file path to `app.py`.
3. In the app's **Settings → Secrets**, add:

```toml
GEMINI_API_KEY = "your_actual_gemini_api_key"
```

4. Save the secrets and deploy/reboot the app. Do not commit the API key to GitHub.
5. Get a key from https://aistudio.google.com/apikey. Gemini model availability and quotas depend on the Google AI Studio project.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

For local secrets, create `.streamlit/secrets.toml` with the same `GEMINI_API_KEY` setting. Do not upload that file to GitHub.
