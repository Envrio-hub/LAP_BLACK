import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('age_and_sex/totals_per_age_and_year.csv', index_col='year')

dependents_group = ['00','01','05','10','15','70','75','80','85','90']
working_age_group = ['20','25','30','35','40','45','50','55','60','65']

depentents = df.loc[:, dependents_group].sum(axis=1)
working_age = df.loc[:, working_age_group].sum(axis=1)
df_dependency_ratio = pd.DataFrame({'Dependents': 100*depentents.values / working_age.values}, index=df.index).round(2)

# Prepare data for regression
X = df_dependency_ratio.index.values.reshape(-1, 1)   # years
y = df_dependency_ratio['Dependents'].values               # values

# Fit linear regression model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Extract parameters
slope = round(model.coef_[0], 2)
intercept = round(model.intercept_, 2)
r2 = round(model.score(X, y),2)

# ---- Plot ----
plt.figure(figsize=(8, 5))
plt.plot(df_dependency_ratio.index, df_dependency_ratio['Dependents'], 'o-', color='teal', label='Observed')
plt.plot(df_dependency_ratio.index, y_pred, '--', color='darkorange', label='Linear fit')

# Annotate statistics
text = f"Slope (a): {slope:.2f}\n$R^2$: {r2:.2f}"#\nIntercept (b): {intercept:.2f}
plt.text(
    0.05, 0.90, text,
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray')
)

# Formatting
plt.xlabel("Year", fontweight='bold', fontsize=14)
plt.ylabel("Dependency Ratio (%)", fontweight='bold', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
# plt.legend()
plt.tight_layout()
plt.show()