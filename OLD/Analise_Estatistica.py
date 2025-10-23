import pandas as pd

df = pd.read_excel('Diferenca_Altimetria.xlsx', skiprows=11, header=1)

print(df.head(100))