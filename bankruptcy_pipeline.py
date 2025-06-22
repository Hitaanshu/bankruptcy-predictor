import pandas as pd
import pickle
import datetime

def run_pipeline():
    with open("log.txt", "a", encoding="utf-8") as log:
        log.write(f"\n[{datetime.datetime.now()}] Pipeline started...\n")

        try:
            with open("final_xgb_model.pkl", "rb") as f:
                model = pickle.load(f)
            df = pd.read_csv("new_data.csv")
            df = df.drop(columns=[col for col in df.columns if col.lower() in ["bankrupt?", "prediction"]], errors="ignore")

            if df.shape[1] != 94:
                raise ValueError("Expected 94 features, got {}".format(df.shape[1]))
            df["Prediction"] = model.predict(df)
            df.to_csv("prediction.csv", index=False)
            log.write(" prediction.csv saved successfully.\n")

        except Exception as e:
            log.write(f"ERROR: {str(e)}\n")

run_pipeline()
