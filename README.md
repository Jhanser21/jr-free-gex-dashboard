# JR GEX / Flow Read Website

A Streamlit dashboard styled like a professional dark gamma terminal.

## Features
- Live underlying price from public market data
- Net, call, and put GEX by strike
- Control node
- Call wall and put wall
- Approximate zero-gamma level
- Bullish, bearish, and chop structures
- Automatic trade-plan paths
- Expected move and max pain
- Responsive website layout

## Deploy on Streamlit Community Cloud
1. Upload all files in this folder to a public GitHub repository.
2. In Streamlit Community Cloud, create a new app.
3. Choose your repository and branch `main`.
4. Set the main file path to `streamlit_app.py`.
5. Deploy.

## Important limitation
The dashboard estimates gamma exposure from public options-chain open interest. It is not a proprietary real-time dealer-positioning feed, and open interest can be delayed.
