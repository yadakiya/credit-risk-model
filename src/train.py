from src.data_processing import preprocess_data


def main():
    df = preprocess_data("data/raw/data.csv")

    print(df.head())
    print(df.shape)


if __name__ == "__main__":
    main()