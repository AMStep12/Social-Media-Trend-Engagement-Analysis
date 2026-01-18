import os, pandas as pd, numpy as np

DATA = "data/synthetic/posts.csv"
OUT = "reports/aggregates"; os.makedirs(OUT, exist_ok=True)

if __name__ == "__main__":
    df = pd.read_csv(DATA, parse_dates=["timestamp"])
    df["engagement"] = df["likes"] + df["comments"] + df["shares"]
    by_topic = df.groupby("topic")["engagement"].mean().sort_values(ascending=False).reset_index()
    by_time = df.set_index("timestamp").resample("D")["engagement"].sum().reset_index()
    by_topic.to_csv(os.path.join(OUT,"avg_engagement_by_topic.csv"), index=False)
    by_time.to_csv(os.path.join(OUT,"engagement_by_day.csv"), index=False)
    print("Wrote dashboard-ready aggregates to reports/aggregates/")
