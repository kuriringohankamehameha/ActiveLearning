# ActiveLearning
An Active Learning library in Python that uses query strategies involving Uncertainty based sampling and Query by Committee for on-line learning.

Uncertainty based sampling methods:
* Least Confident
* Maximum Margin

Query by Committee methods:
* Vote Entropy
* KL Divergence

This also tries to minimize the budget for labeling from the Oracle wherever possible, and uses the reduction of total distance of K Nearest Neighbours to achieve this task.
