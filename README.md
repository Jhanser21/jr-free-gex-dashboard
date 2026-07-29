# JR Free GEX Dashboard

A free Streamlit dashboard that estimates gamma exposure from public options-chain data.

## Website deployment

1. Create a new GitHub repository named `jr-free-gex-dashboard`.
2. Upload the **contents of this folder** to the repository root:
   - `streamlit_app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
3. Open Streamlit Community Cloud and sign in with GitHub.
4. Select **Create app** → **Yup, I have an app**.
5. Select your repository and use `streamlit_app.py` as the entrypoint.
6. Choose a custom subdomain such as `jr-free-gex-dashboard`, if available.
7. Click **Deploy**.

## Notes

- Public option-chain data may be delayed and open interest is generally not intraday real-time.
- GEX is an estimate. Exact dealer positioning is not publicly known.
- The app caches data for five minutes. Use the Refresh button to clear the cache.
