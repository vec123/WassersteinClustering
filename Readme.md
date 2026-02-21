The code for a Wasserstein clusering approach.
Important hyperparameters are the window lenght.
Considers each window as a 1D distribution, represented by the histogram of percentages,
i.e. (-10%, -5%, 0%, 5%, 10%). 
Similar distributions go to the same cluster with similarity/geometry defined by the wasserstein distance/metric.
