from flask import Flask, render_template, request, jsonify, make_response
import threading
import webbrowser

# Import our custom modules
from stock_data import fetch_historical_data, fetch_market_overview, generate_csv_data, get_usd_inr_rate
from model import train_and_predict

app = Flask(__name__)

@app.route('/')
def index():
    # Render HTML from templates/index.html
    return render_template('index.html')

@app.route('/market-overview')
def market_overview():
    try:
        stocks = fetch_market_overview()
        return jsonify({'stocks': stocks})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download-stocks')
def download_stocks():
    try:
        csv_string = generate_csv_data()
        output = make_response(csv_string)
        output.headers["Content-Disposition"] = "attachment; filename=market_data_all_companies.csv"
        output.headers["Content-type"] = "text/csv"
        return output
    except Exception as e:
        return f"Error assembling CSV: {str(e)}", 500

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    ticker = data.get('ticker', 'AAPL')
    
    try:
        # 1. Get Live Historical Data
        stock = fetch_historical_data(ticker)
        
        if stock is None or stock.empty:
            return jsonify({'error': 'No data found for this ticker.'}), 404
            
        if len(stock) < 60:
             return jsonify({'error': 'Not enough historical data to predict.'}), 400

        # 2. Get Machine Learning Prediction
        predicted_price = train_and_predict(stock, forecast_out=30)
        
        current_price = float(stock['Close'].iloc[-1])
        
        # 3. Format Data for the UI Chart
        recent_data = stock.tail(252)
        dates = [d.strftime('%Y-%m-%d') for d in recent_data.index]
        prices = [float(p) for p in recent_data['Close'].values]
        
        # Currency Conversion: If it's a US stock, convert everything to INR
        if not ticker.endswith('.NS') and not ticker.endswith('.BO'):
            rate = get_usd_inr_rate()
            predicted_price *= rate
            current_price *= rate
            prices = [p * rate for p in prices]
            
        return jsonify({
            'historical': {
                'dates': dates,
                'prices': prices
            },
            'current_price': current_price,
            'predicted_price': predicted_price
        })
        
    except Exception as e:
        return jsonify({'error': f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    print("=========================================================")
    print("Starting AI Stock Prediction Dashboard")
    print("Access your dashboard at http://127.0.0.1:5000")
    print("=========================================================")
    
    def open_browser():
        webbrowser.open_new("http://127.0.0.1:5000")
        
    # Start the browser automatically
    threading.Timer(1.5, open_browser).start()
    
    app.run(debug=True, port=5000)
