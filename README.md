# Stock Return Prediction and Portfolio Construction Project

## Project Overview
This project implements a machine learning-based approach to predict stock returns and construct profitable long-short portfolios. The system uses various machine learning models, including XGBoost, Lasso, Ridge, and ElasticNet, to predict stock returns and generate trading signals.

## Project Structure
```
.
├── data/               # Raw and processed data files
├── models/            # Trained model files
├── reports/           # Analysis reports and documentation
├── figures/           # Generated visualizations
├── plots/             # Additional plots and charts
├── src/               # Source code
└── venv/              # Python virtual environment
```

## Key Components

### 1. Data Processing
- Raw data processing and feature engineering
- Target variable transformation analysis
- Feature distribution analysis
- Data quality checks and preprocessing

### 2. Model Development
- Baseline models (Lasso, Ridge, ElasticNet)
- Advanced XGBoost model
- Model performance comparison
- Feature importance analysis

### 3. Portfolio Construction
- Long-short portfolio strategy
- Position sizing and risk management
- Portfolio rebalancing
- Performance tracking

### 4. Performance Evaluation
- Return metrics (Sharpe ratio, annualized returns)
- Risk metrics (volatility, drawdowns)
- Transaction costs analysis
- Performance attribution

## Results

### Model Performance
- XGBoost achieved superior performance compared to baseline models
- Strong out-of-sample predictive power
- Robust feature importance rankings

### Portfolio Performance
- Average Monthly Return: 7.03%
- Sharpe Ratio: 2.58
- Consistent positive performance across market conditions
- Effective risk management through diversification

## Usage

### Setup
1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Pipeline
1. Data Processing:
```bash
python src/data_processing.py
```

2. Model Training:
```bash
python src/train_models.py
```

3. Portfolio Construction:
```bash
python src/construct_portfolio.py
```

4. Performance Evaluation:
```bash
python src/performance.py
```

## Documentation
Detailed documentation can be found in the `reports/` directory:
- `data_summary.txt`: Data overview and statistics
- `feature_distribution_analysis.txt`: Feature analysis
- `target_analysis.txt`: Target variable analysis
- `performance_table.csv`: Model performance metrics
- `top_10_holdings.md`: Portfolio composition analysis

## Future Improvements
1. Implement additional machine learning models
2. Enhance feature engineering
3. Optimize portfolio construction methodology
4. Add real-time trading capabilities
5. Implement more sophisticated risk management

## Contributing
Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details. 