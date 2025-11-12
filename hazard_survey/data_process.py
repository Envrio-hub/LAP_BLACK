import pandas as pd, numpy as np

data = pd.read_csv('hazards_questionnaires.csv', sep=';')

participants = data['participant'].unique()

rows = data[data['participant']==participants[0]]['row'].unique()

columns = data[(data['participant']==participants[0])&(data['row']==rows[0])]['col'].unique()

matrices_list = []

for participant in participants:
    zeros_dict = {}
    for col in columns:
        data_slice = np.array(data[(data['participant']==participant)&(data['col']==col)]['value'])
        if len(data_slice)!=len(rows):
            filler_array = ([0]*(len(rows)-len(data_slice)))
            for element in filler_array:
                data_slice = np.append(data_slice, element)
        zeros_dict[col] = data_slice
    df_data = pd.DataFrame.from_dict(zeros_dict, orient='columns')
    df_data.index = rows
    # Copy to avoid modifying original directly
    df_filled = df_data.copy().astype(float)

    # Iterate over lower-triangular positions (i > j)
    for i in range(df_filled.shape[0]):
        for j in range(df_filled.shape[1]):
            if i > j:  # below diagonal
                df_filled.iloc[i, j] = 1 / df_filled.iloc[j, i]

    matrices_list.append(df_filled)

# --- Group aggregation (geometric mean) ---
A_group = np.exp(np.mean([np.log(A) for A in matrices_list], axis=0))

# Row geometric means
g = np.prod(A_group, axis=1)**(1/A_group.shape[0])
w = g / g.sum()

# Consistency ratio
Aw = A_group @ w
lambda_max = np.mean(Aw / w)
n = A_group.shape[0]
CI = (lambda_max - n) / (n - 1)

# Random Index (Saaty)
RI = {1:0.00, 2:0.00, 3:0.58, 4:0.90, 5:1.12, 6:1.24, 7:1.32, 8:1.41, 9:1.45, 10:1.49}[n]
CR = CI / RI if RI > 0 else 0

print(f"Weights: {w.round(3)}")
print(f"λmax={lambda_max:.3f}, CI={CI:.3f}, CR={CR:.3f}")

print()