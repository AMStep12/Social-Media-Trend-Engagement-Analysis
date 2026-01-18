import os, numpy as np, pandas as pd

np.random.seed(7)
topics = ["ai","sports","gaming","finance","fashion","food","travel","health"]
users = [f"user_{i}" for i in range(500)]

def synth(n=50000):
    t = np.random.choice(topics, size=n, p=[0.2,0.15,0.12,0.1,0.1,0.1,0.1,0.13])
    length = np.random.poisson(18, size=n) + 3
    base_eng = np.random.gamma(2, 5, size=n)
    sentiment = np.clip(np.random.normal(loc=[0.6 if x in ["ai","sports","gaming"] else 0.4][0] if isinstance(x,str) else 0.5, scale=0.3), -1, 1)
    # build text and engagement
    text = [f"{ti} post " + " ".join(["word"]*int(l)) for ti, l in zip(t, length)]
    likes = (base_eng * (1 + (np.array([0.2 if x in ['ai','sports'] else 0 for x in t])))).astype(int)
    comments = (likes * np.random.uniform(0.05, 0.2, size=n)).astype(int)
    shares = (likes * np.random.uniform(0.03, 0.15, size=n)).astype(int)
    df = pd.DataFrame({
        "user": np.random.choice(users, size=n),
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="H"),
        "topic": t,
        "text": text,
        "likes": likes,
        "comments": comments,
        "shares": shares
    })
    return df

if __name__ == "__main__":
    outdir = "data/synthetic"
    os.makedirs(outdir, exist_ok=True)
    df = synth(20000)
    df.to_csv(os.path.join(outdir, "posts.csv"), index=False)
    print(f"Wrote {len(df):,} rows to {os.path.join(outdir,'posts.csv')}")
