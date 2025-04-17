import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("example_data/clustered_points.csv")

plt.figure(figsize=(8, 5.5))

# Plot each cluster separately
for cluster_id in sorted(df['cluster'].unique()):
    subset = df[df['cluster'] == cluster_id]
    label = f"Cluster {cluster_id}" if cluster_id >= 0 else "Noise"
    plt.scatter(subset['x'], subset['y'], s=10, label=label)

plt.title("DBSCAN Clustering Results")
plt.xlabel("x")
plt.ylabel("y")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/dbscan_clusters_visualized.png", 
            dpi=300, bbox_inches='tight', facecolor='white')
# plt.show()
