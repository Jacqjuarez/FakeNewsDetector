import pandas as pd

fake = pd.read_csv("data/Fake.csv")
true = pd.read_csv("data/True.csv")

fake["label"] = "FAKE"
true["label"] = "REAL"

df = pd.concat([fake, true], axis=0).sample(frac=1).reset_index(drop=True)
df.to_csv("data/news.csv", index=False)
print("✅ news.csv created successfully!")
