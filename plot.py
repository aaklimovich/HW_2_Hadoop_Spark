import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/all_results.csv")

df["label"] = df["nodes"].astype(str) + " nodes, opt=" + df["optimized"].astype(str)

x = range(len(df))

plt.figure()
plt.bar(x, df["time"])
plt.xticks(x, df["label"], rotation=30)
plt.title("Execution Time")
plt.xlabel("Configuration")
plt.ylabel("Time")
for i, v in enumerate(df["time"]):
    plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig("time.png")

plt.figure()
plt.bar(x, df["ram"])
plt.xticks(x, df["label"], rotation=30)
plt.title("RAM Usage")
plt.xlabel("Configuration")
plt.ylabel("RAM (MB)")
for i, v in enumerate(df["ram"]):
    plt.text(i, v, f"{v:.2f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig("ram.png")

plt.figure()
plt.bar(x, df["auc"])
plt.xticks(x, df["label"], rotation=30)
plt.title("AUC Score")
plt.xlabel("Configuration")
plt.ylabel("AUC")
for i, v in enumerate(df["auc"]):
    plt.text(i, v, f"{v:.4f}", ha='center', va='bottom')
plt.tight_layout()
plt.savefig("auc.png")

plt.show()