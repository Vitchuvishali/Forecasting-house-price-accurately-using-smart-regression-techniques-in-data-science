# House Price Prediction

A machine learning project that predicts house prices based on property features such as area, bedrooms, bathrooms, floors, year built, location, condition, and garage availability. The project includes data exploration, preprocessing, model building, and an interactive Gradio web app for live predictions.

## Dataset

The dataset (`House Price Prediction Dataset.csv`) contains 2000 records with the following columns:

| Column | Description |
|---|---|
| Id | Unique identifier for each record |
| Area | House area in square feet |
| Bedrooms | Number of bedrooms |
| Bathrooms | Number of bathrooms |
| Floors | Number of floors |
| YearBuilt | Year the house was built |
| Location | Downtown / Suburban / Urban / Rural |
| Condition | Excellent / Good / Fair / Poor |
| Garage | Yes / No |
| Price | Target variable — sale price of the house |

No missing values or duplicate rows were found in the dataset.

## Project Workflow

1. **Data Loading** — Load the CSV file using pandas.
2. **Data Exploration** — Inspect shape, columns, data types, and summary statistics.
3. **Missing Value & Duplicate Check** — Confirm data quality.
4. **Visualization** — Histograms (Area, Price), scatter plot (Area vs Price by Location), and box plot (Price by Location) using Seaborn/Matplotlib.
5. **Feature Engineering**
   - Separate target (`Price`) and features.
   - One-hot encode categorical columns (`Location`, `Condition`, `Garage`) using `pd.get_dummies`.
   - Scale numerical features with `MinMaxScaler`.
6. **Train-Test Split** — 80/20 split (1600 train / 400 test records).
7. **Model Building**
   - Baseline model: `LinearRegression`.
   - Deployment model: `RandomForestRegressor` (trained on label-encoded categorical features).
8. **Evaluation** — R² Score and Mean Absolute Error (MAE).
9. **Deployment** — An interactive **Gradio** web app where users can input house details and get an estimated price.

## Results

| Metric | Value |
|---|---|
| R² Score | ~0.008 |
| MAE | ~237,786 |

> Note: The low R² score indicates that, for this synthetic dataset, the chosen features have weak linear correlation with price. Further feature engineering or alternate models (e.g., Random Forest, Gradient Boosting) may improve performance.

## Tech Stack

- Python 3
- pandas, numpy
- scikit-learn (LinearRegression, RandomForestRegressor, MinMaxScaler, LabelEncoder)
- seaborn, matplotlib (visualization)
- Gradio (web app deployment)

## Installation

```bash
git clone https://github.com/<your-username>/house-price-prediction.git
cd house-price-prediction
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
scikit-learn
seaborn
matplotlib
gradio
```

## Usage

1. Place `House Price Prediction Dataset.csv` in the project directory.
2. Run the notebook or script:

```bash
python app.py
```

3. Open the local Gradio URL printed in the terminal (a public shareable link is also generated automatically).
4. Enter property details (Area, Bedrooms, Bathrooms, Floors, Year Built, Location, Condition, Garage) and click **Submit** to get the estimated price.

## App Preview

The Gradio interface ("House Price Predictor") takes the following inputs:
- Area (sq ft)
- Bedrooms
- Bathrooms
- Floors
- Year Built
- Location (dropdown)
- Condition (dropdown)
- Garage (dropdown)

And returns an estimated price, e.g.:
```
Estimated Price: $532,981
```

## Project Structure

```
house-price-prediction/
│
├── House Price Prediction Dataset.csv   # Dataset
├── app.py                                # Model training + Gradio app
├── README.md                              # Project documentation
└── requirements.txt                       # Dependencies
```

## Future Improvements

- Engineer additional features (e.g., house age, price per sq ft).
- Try advanced regression models (XGBoost, Gradient Boosting) and hyperparameter tuning.
- Add cross-validation for more robust evaluation.
- Persist the trained model with `joblib`/`pickle` for faster reloads.
- Deploy permanently via Hugging Face Spaces or Render instead of a temporary Gradio share link.

## License

This project is open-source and available under the MIT License.
