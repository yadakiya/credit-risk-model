from src.data_processing import preprocess_data, build_rfm_target


def main():
    path = "data/raw/data.csv"

    # STEP 1: load clean raw data
    df = preprocess_data(path)

    print("Processed data shape:", df.shape)

    # STEP 2: build RFM target (uses RAW df, NOT aggregated)
    rfm_target = build_rfm_target(df)

    print("\nRFM TARGET SAMPLE:")
    print(rfm_target.head())


if __name__ == "__main__":
    main()
