The code for a Wasserstein clusering approach of timeseries.

The main important hyperparameter is the window length.
Considers each window as a 1D distribution, represented by the histogram of percentages,
i.e. (-10%, -5%, 0%, 5%, 10%). 
Similar distributions go to the same cluster with similarity/geometry defined by the wasserstein distance/metric.

I got interested in the Wasserstein distance while working for a VAE for Shapes, represented as Signed Distance functions.
Turns out the MSE and proportional distances have some disadvantages when working with few shapes. 
The Wasserstein distance, based on optimal (mass) transport, would be a much nicer
metric for quantifying when two shapes are similar.
Sadly, in high-dimensions, it is rather expensive to compute.


This approach applies it to 1D time-series. For the 1D computation is very cheap.
In fact, an approach for multi-dim. Wasserstein distance computation uses the Radon transform (a method used often in tomography) to project the distribution onto 1D slices
( see sliced Wasserstein distance)