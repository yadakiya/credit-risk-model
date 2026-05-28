from src.data_processing import preprocess_data

df = preprocess_data("data/raw/data.csv")

print(df.head())
print(df.shape)