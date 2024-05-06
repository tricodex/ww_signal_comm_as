import statsmodels.api as sm
from statsmodels.formula.api import ols
import pandas as pd
import pickle

# Load the data
try:
    with open('reward_data.pkl', 'rb') as f:
        loaded_rewards = pickle.load(f)
except FileNotFoundError:
    print("File not found. Please check the file path and name.")
    exit(1)

# Validate data structure
if not isinstance(loaded_rewards, dict) or not all(isinstance(v, list) for v in loaded_rewards.values()):
    print("Data structure is incorrect. Expected a dictionary of lists.")
    exit(1)

# Create a DataFrame
df = pd.DataFrame(loaded_rewards)

# Check for the expected number of configurations
expected_columns = [f'config_{i+1}' for i in range(12)]
if not all(col in df.columns for col in expected_columns):
    print("Missing some configurations in the data. Please check the data integrity.")
    exit(1)

# Melt the DataFrame to 'long-form' for ANOVA
df_melt = pd.melt(df.reset_index(), id_vars=['index'], value_vars=expected_columns)
df_melt.columns = ['index', 'treatments', 'value']

# Perform ANOVA
model = ols('value ~ C(treatments)', data=df_melt).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
