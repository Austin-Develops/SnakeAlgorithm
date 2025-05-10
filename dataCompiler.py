import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load data
with open('results copy.json', 'r') as f:
    data = json.load(f)

# Convert JSON to a long-form DataFrame
records = []
for agent, tests in data.items():
    for test_name, results in tests.items():
        records.append({
            "Agent": agent,
            "Test": test_name,
            "Apples Eaten": results["Apples Eaten"],
            "Time": results["Time"]
        })

df = pd.DataFrame(records)

# Set plot style
sns.set(style="whitegrid")

# Create box plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot for Apples Eaten
sns.boxplot(data=df, x="Agent", y="Apples Eaten", ax=axes[0])
axes[0].set_title("Apples Eaten per Agent")
axes[0].tick_params(axis='x', rotation=45)

# Box plot for Time
sns.boxplot(data=df, x="Agent", y="Time", ax=axes[1])
axes[1].set_title("Time per Agent")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()
