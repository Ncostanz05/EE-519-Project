import pandas as pd

df = pd.read_csv("commonvoice-v24_en-AU/commonvoice-v24_en-AU.csv")
print(df["age"].value_counts())