# Matt Lewis
# ECU Data Mining - Homework 2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

WINDOW_SIZE = 1000  # Chosen window size W
PERCENTILE_Q = 84.5  # Percentile threshold (upper tail)
DATA_FILE = 'AG_NO3_fill_cells_remove_NAN-2.csv'
OUTPUT_PLOT = 'imgs/anomaly_detection_results.png'

# Loading data
df = pd.read_csv(DATA_FILE)

# Extract Nitrate values and ground truth flags
nitrate = df['NO3N'].values
ground_truth = df['Student_Flag'].values  # 1 = anomaly, 0 = normal

n_points = len(nitrate)

# Initialize predictions array
predictions = np.zeros(n_points, dtype=int)  # 0 = normal, 1 = anomaly

# Sliding window anomaly detection
print(f"Running sliding window detection (W={WINDOW_SIZE}, q={PERCENTILE_Q})...")

# Window 1: Label all points in the first window
window_data = nitrate[0:WINDOW_SIZE]
threshold = np.percentile(window_data, PERCENTILE_Q, method='linear')
for i in range(WINDOW_SIZE):
    if nitrate[i] >= threshold:
        predictions[i] = 1

# Subsequent windows: Label only the newly added point
for i in range(1, n_points - WINDOW_SIZE + 1):
    window_start = i
    window_end = i + WINDOW_SIZE
    
    window_data = nitrate[window_start:window_end]
    threshold = np.percentile(window_data, PERCENTILE_Q, method='linear')
    
    # Label only the newly added point (last point in window)
    new_point_idx = window_end - 1
    if nitrate[new_point_idx] >= threshold:
        predictions[new_point_idx] = 1

# Evaluation Metrics
print("Evaluation Metrics")
print("="*50)

# Confusion matrix components
TP = np.sum((predictions == 1) & (ground_truth == 1)) # True Positive
FP = np.sum((predictions == 1) & (ground_truth == 0)) # False Positive
FN = np.sum((predictions == 0) & (ground_truth == 1)) # False Negative
TN = np.sum((predictions == 0) & (ground_truth == 0)) # True Negative

# Totals
P = np.sum(ground_truth == 1) # Total anomalies
N = np.sum(ground_truth == 0) # Total normals

# Accuracy metrics
normal_accuracy = (TN / N) * 100 if N > 0 else 0
anomaly_accuracy = (TP / P) * 100 if P > 0 else 0

print(f"\nConfusion Matrix:")
print(f"  TP (True Positive):  {TP}")
print(f"  FP (False Positive): {FP}")
print(f"  FN (False Negative): {FN}")
print(f"  TN (True Negative):  {TN}")

print(f"\nTotals:")
print(f"  P (Total Anomalies): {P}")
print(f"  N (Total Normals):   {N}")

print(f"\nAccuracy Metrics:")
print(f"  Normal Event Detection Accuracy:  {normal_accuracy:.2f}% (TN/N)")
print(f"  Anomaly Event Detection Accuracy: {anomaly_accuracy:.2f}% (TP/P)")

print(f"\nTarget Performance:")
print(f"  Normal accuracy ≥ 80%:  {'✓ PASS' if normal_accuracy >= 80 else '✗ FAIL'}")
print(f"  Anomaly accuracy ≥ 75%: {'✓ PASS' if anomaly_accuracy >= 75 else '✗ FAIL'}")

# Event-level anomaly evaluation
def count_events(labels):
    """Count contiguous runs of anomalies (1s) as single events."""
    events = 0
    in_event = False
    for val in labels:
        if val == 1 and not in_event:
            events += 1
            in_event = True
        elif val == 0:
            in_event = False
    return events

gt_events = count_events(ground_truth)
pred_events = count_events(predictions)

print(f"\nEvent-Level Analysis:")
print(f"  Ground truth anomaly events: {gt_events}")
print(f"  Predicted anomaly events:    {pred_events}")

# Visualization
plt.figure(figsize=(16, 6))

# Plot normal points
normal_mask = (predictions == 0) & (ground_truth == 0)
plt.scatter(np.where(normal_mask)[0], nitrate[normal_mask], 
            c='blue', s=10, alpha=0.5, label='Normal')

# Plot detected anomalies
detected_mask = predictions == 1
plt.scatter(np.where(detected_mask)[0], nitrate[detected_mask], 
            c='red', marker='x', s=50, alpha=0.7, label='Detected Anomaly')

# Plot ground truth anomalies
gt_anomaly_mask = ground_truth == 1
plt.scatter(np.where(gt_anomaly_mask)[0], nitrate[gt_anomaly_mask], 
            facecolors='none', edgecolors='orange', s=100, linewidths=2, 
            label='Ground Truth Anomaly')

plt.xlabel('Time Index', fontsize=12)
plt.ylabel('Nitrate (NO3-N)', fontsize=12)
plt.title(f'Anomaly Detection Results (W={WINDOW_SIZE}, q={PERCENTILE_Q}%)\n'
          f'Normal Acc: {normal_accuracy:.2f}%, Anomaly Acc: {anomaly_accuracy:.2f}%', 
          fontsize=14, fontweight='bold')
plt.legend(loc='upper right', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{OUTPUT_PLOT}'")

plt.show()

print("\nAnalysis complete.")
