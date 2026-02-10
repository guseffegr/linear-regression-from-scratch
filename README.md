# Linear Regression from Scratch

This project implements linear regression from scratch using NumPy.
The goal is to understand the full machine learning pipeline — from data preprocessing
and feature engineering to model evaluation and comparison with scikit-learn.

## Dataset

The project uses the **California Housing** dataset from scikit-learn.

- Each row represents aggregated housing statistics for a census block group in California
- The target variable is **MedHouseVal** — median house value
- All features are numerical, which makes the dataset suitable for linear regression

## Project structure

├── src/
│   ├── model.py          # Linear regression implementation from scratch
│   ├── preprocessing.py # Train/val/test split and normalization
│   ├── loss.py           # RMSE metric
│   └── plotting.py       # Visualization utilities
├── notebooks/
│   └── linear_regression_from_scratch.ipynb
└── README.md

## Methodology

The project follows a standard machine learning workflow:

1. Data loading and inspection
2. Train / validation / test split
3. Z-score normalization (computed on training set only)
4. Training a baseline linear regression model using batch gradient descent
5. Feature–target analysis to identify non-linear patterns
6. Feature engineering to improve model expressiveness
7. Model evaluation using RMSE
8. Comparison with scikit-learn implementation

## Feature engineering

Several engineered features are introduced to better capture non-linear relationships:

- Polynomial terms for latitude and longitude
- Interaction term between latitude and longitude
- Proxy for distance to the coast derived from longitude
- Square root transformation of median income
- Population density feature combining population and average occupancy
- Bedrooms-to-rooms ratio as a structural housing quality indicator

## Regularization

L2 regularization (Ridge) was evaluated using multiple values of the regularization
parameter λ. Validation results show that regularization does not improve performance
for this dataset after feature engineering.

The best validation RMSE is achieved with λ = 0, indicating that the model is not
overfitting and that feature engineering is the dominant factor.

## Results

| Model | Train RMSE | Validation RMSE | Test RMSE |
|------|-----------|-----------------|-----------|
| Scratch (GD) | ~0.688 | ~0.712 | ~0.681 |
| Scikit-learn | ~0.675 | ~0.704 | ~0.666 |

## Visualizations

The notebook includes:
- Training loss (RMSE) vs iterations
- Feature vs target scatter plots
- Predicted vs actual values on the test set

## Conclusion

The project demonstrates that a manually implemented linear regression model can
achieve performance close to a production-grade library when preprocessing,
feature engineering, and training are done correctly.

While scikit-learn provides better numerical efficiency, the scratch implementation
offers full transparency and educational value.

## How to run

1. Clone the repository
2. Install dependencies: `numpy, pandas, matplotlib, scikit-learn`
3. Run the notebook in `notebooks/`