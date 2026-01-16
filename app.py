print("APP FILE STARTED")

from flask import Flask, render_template, request

from train_models import (
    fetch_stock_data,
    linear_regression_prediction,
    lstm_prediction,
    plot_stock_with_prediction
)

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    company = request.form.get("company")
    model = request.form.get("model")

    if not company or not model:
        return render_template(
            "index.html",
            prediction="Please select both company and model."
        )

    ticker_map = {
        "apple": "AAPL",
        "microsoft": "MSFT",
        "google": "GOOGL",
        "tesla": "TSLA",
        "zomato": "ZOMATO.NS"
    }

    ticker = ticker_map.get(company)
    if ticker is None:
        return render_template(
            "index.html",
            prediction="Invalid company selection."
        )

    df = fetch_stock_data(ticker)
    if df.empty:
        return render_template(
            "index.html",
            prediction="Stock data unavailable."
        )

    if model == "linear":
        future_prices = linear_regression_prediction(df, years=3)
        model_name = "Linear Regression"

    elif model == "lstm":
        future_prices = lstm_prediction(df, years=3)
        model_name = "LSTM"

    else:
        return render_template(
            "index.html",
            prediction="Invalid model selection."
        )

    plot_stock_with_prediction(df, future_prices, company, model_name)

    prediction_text = (
        f"{company.capitalize()} prediction using {model_name} "
        f"(3 years ahead)"
    )

    return render_template(
        "index.html",
        chart="stock_plot.png",
        prediction=prediction_text
    )


if __name__ == "__main__":
    print("STARTING FLASK SERVER")
    app.run(debug=True)

