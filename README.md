By Benjamin Z.Y. Teoh @ July 2024 @ Alpharetta, GA

## Overview
This application visualizes the implied volatility of an asset using its call and put options. It provides insight into how these volatilities vary with strike price and expiration date. While theoretical models sometimes simplify volatility surfaces, actual market data shows that they are more complex. This application plots the empirical volatility surfaces derived from both call and put options data, offering a more accurate representation of market conditions.

### How to Use:
1. Select or enter the ticker symbol of interest. The dropdown menu consists of all 30 tickers from the Dow Jones Industrial Average.
2. Choose between a 1/2-year or 1-year period for options contracts with expiration dates within the selected range.
3. Click "Go" to generate plots showing volatility versus strike price and expiration date. The "Download CSV" saves the _options_ data to your device.

### Important Notes:
1. Data is sourced from Yahoo Finance, so options data for non-US markets may not be available.
2. The implied volatility values are directly from Yahoo Finance.
3. The plotted surfaces are regression surfaces based on the implied volatility data.
4. Volatility values close to zero (specifically, < 0.01) are considered erroneous and excluded from the regression analysis, though they are still shown on the plots.

If you have any questions or comments, email me at work.teohzuyao@gmail.com.
