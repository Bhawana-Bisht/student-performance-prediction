import pandas as pd
import matplotlib.pyplot as plt
import os

# Ensure reports folder exists
os.makedirs("reports", exist_ok=True)

df = pd.read_csv("data/raw/students.csv")

plt.figure()
plt.hist(df["final_score"], bins=20)
plt.title("Student Performance Distribution")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")

plt.savefig("reports/performance_distribution.png")
plt.close()

print("Report generated: reports/performance_distribution.png")
