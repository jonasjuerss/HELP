To reproduce the experiments in the paper, please use `python train.py` with the given arguments and seeds 1-3

## Hierarchical Synthetic
### Ours
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"KMeans\",\"num_concepts\":10,\"global_clusters\":false,\"kmeans_threshold\":0.02}' '{\"cluster_alg\":\"KMeans\",\"num_concepts\":15,\"global_clusters\":false,\"kmeans_threshold\":0.02}' '{}' --num_epochs 10000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --wandb_name 'hierarch-kmeans10_15_2%' --wandb_tags final kmeans-merge --seed
```
### GCN
```
--conv_type DenseGCNConv --dense_data --pooling_type DenseNoPool --pool_blocks '{}' --num_epochs 10000 --add_layer 32 32 32 32 32 4 --output_layer_merge sum --wandb_name 'hierarch-gcn' --wandb_tags final gcn --seed
```
### DiffPool
```
--conv_type DenseGCNConv --dense_data --pooling_type DiffPool DiffPool DenseNoPool --pool_blocks '{\"num_output_nodes\":10}' '{\"num_output_nodes\":5}' '{}' --num_epochs 10000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --pooling_loss_weight 1 --wandb_name 'hierarch-diffpool10_5' --wandb_tags final diffpool --seed 
```
### ASAP
```
--conv_type GCNConv --sparse_data --pooling_type ASAP ASAP SparseNoPool --pool_blocks '{\"ratio_output_nodes\":0.5}' '{\"ratio_output_nodes\":0.5}' '{}' --num_epochs 10000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --wandb_name 'hierarch-asap50%' --wandb_tags final asap --seed
```
### Hyperplane
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"KMeans\",\"num_concepts\":10,\"global_clusters\":false,\"kmeans_threshold\":0.02,\"num_mc_samples\":10,\"perturbation\":{\"_type\":\"GaussianPerturbation\",\"args\":{\"std\":0.1}}}' '{\"cluster_alg\":\"KMeans\",\"num_concepts\":15,\"global_clusters\":false,\"kmeans_threshold\":0.02,\"num_mc_samples\":10,\"perturbation\":{\"_type\":\"GaussianPerturbation\",\"args\":{\"std\":0.1}}}' '{}' --num_epochs 10000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --wandb_name 'hierarch-hyperplane10_0.3-kmeans10_15_2%' --blackbox_transparency 0.3 --wandb_tags final hyperplane-kmeans-merge --seed
```
### Global
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"KMeans\",\"num_concepts\":10,\"global_clusters\":true,\"kmeans_threshold\":0.0}' '{\"cluster_alg\":\"KMeans\",\"num_concepts\":15,\"global_clusters\":true,\"kmeans_threshold\":0.0}' '{}' --num_epochs 10000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --wandb_name 'hierarch-kmeans10_15_global' --wandb_tags final kmeans-global --seed
```
## Mutagenicity
### Ours
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"KMeans\",\"num_concepts\":20,\"global_clusters\":false,\"kmeans_threshold\":0.02}' '{\"cluster_alg\":\"KMeans\",\"num_concepts\":20,\"global_clusters\":false,\"kmeans_threshold\":0.02}' '{}' --dataset '{\"args\":{},\"_type\":\"MutagenicityWrapper\"}' --num_epochs 5000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --wandb_name 'mutag-kmeans20_20_2%' --wandb_tags final kmeans-merge --seed
```

### Hyperplane
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"KMeans\",\"num_concepts\":20,\"global_clusters\":false,\"kmeans_threshold\":0.02,\"num_mc_samples\":10,\"perturbation\":{\"_type\":\"GaussianPerturbation\",\"args\":{\"std\":0.1}}}' '{\"cluster_alg\":\"KMeans\",\"num_concepts\":20,\"global_clusters\":false,\"kmeans_threshold\":0.02,\"num_mc_samples\":10,\"perturbation\":{\"_type\":\"GaussianPerturbation\",\"args\":{\"std\":0.1}}}' '{}' --dataset '{\"args\":{},\"_type\":\"MutagenicityWrapper\"}' --blackbox_transparency 0.3 --num_epochs 5000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --wandb_name 'mutag-kmeans20_20_2%' --wandb_tags final hyperplane-kmeans-merge --seed
```

### GCN
```
--conv_type DenseGCNConv --dense_data --pooling_type DenseNoPool --pool_blocks '{}' --dataset '{\"args\":{},\"_type\":\"MutagenicityWrapper\"}' --num_epochs 5000 --add_layer 32 32 32 32 32 4 --output_layer_merge sum --wandb_name 'mutag-gcn' --wandb_tags final gcn --seed 
```
### DiffPool
```
--conv_type DenseGCNConv --dense_data --pooling_type DiffPool DiffPool DenseNoPool --pool_blocks '{\"num_output_nodes\":10}' '{\"num_output_nodes\":5}' '{}' --dataset '{\"args\":{},\"_type\":\"MutagenicityWrapper\"}' --num_epochs 5000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --pooling_loss_weight 1 --wandb_name 'mutag-diffpool10_5' --wandb_tags final diffpool --seed 
```

### ASAP
```
--conv_type GCNConv --sparse_data --pooling_type ASAP ASAP SparseNoPool --pool_blocks '{\"ratio_output_nodes\":0.5}' '{\"ratio_output_nodes\":0.5}' '{}' --dataset '{\"args\":{},\"_type\":\"MutagenicityWrapper\"}' --num_epochs 5000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --wandb_name 'mutag-asap50%' --wandb_tags final asap --seed
```
### Global
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"KMeans\",\"num_concepts\":20,\"global_clusters\":true,\"kmeans_threshold\":0.0}' '{\"cluster_alg\":\"KMeans\",\"num_concepts\":20,\"global_clusters\":true,\"kmeans_threshold\":0.0}' '{}' --dataset '{\"args\":{},\"_type\":\"MutagenicityWrapper\"}' --num_epochs 5000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --wandb_name 'mutag-kmeans20_20_global' --wandb_tags final kmeans-global --seed
```
### Sequential
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"SequentialKMeansMeanShift\",\"global_clusters\":false,\"num_sketches\":60,\"mean_shift_range\":0.02,\"min_samples_per_sketch\":0.001,\"cluster_decay_factor\":0.9,\"rescale_clusters_decay\":0.8}' '{\"cluster_alg\":\"SequentialKMeansMeanShift\",\"global_clusters\":false,\"num_sketches\":60,\"mean_shift_range\":0.02,\"min_samples_per_sketch\":0.001,\"cluster_decay_factor\":0.9,\"rescale_clusters_decay\":0.8}' '{}' --dataset '{\"args\":{},\"_type\":\"MutagenicityWrapper\"}' --num_epochs 5000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --wandb_name 'mutag-sequentialkmeans' --wandb_tags final sequentialkmeans-merge --seed 
```
### MeanShift
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"MeanShift\",\"num_concepts\":20,\"global_clusters\":false,\"kmeans_threshold\":0.02}' '{\"cluster_alg\":\"MeanShift\",\"num_concepts\":20,\"global_clusters\":false,\"kmeans_threshold\":0.02}' '{}' --dataset '{\"args\":{},\"_type\":\"MutagenicityWrapper\"}' --num_epochs 5000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --wandb_name 'mutag-kmeans20_20_2%' --wandb_tags final kmeans-merge --seed 
```

## Reddit Binary
### Ours
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"KMeans\",\"num_concepts\":30,\"global_clusters\":false,\"kmeans_threshold\":0.02}' '{\"cluster_alg\":\"KMeans\",\"num_concepts\":30,\"global_clusters\":false,\"kmeans_threshold\":0.02}' '{}' --num_epochs 3500 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --dataset '{\"args\":{},\"_type\":\"RedditBinaryWrapper\"}' --output_layer_merge sum --wandb_name 'redditbin-kmeans30_30_2%' --wandb_tags final kmeans-merge --batch_size 32 --seed 
```
### GCN
```
--conv_type DenseGCNConv --dense_data --pooling_type DenseNoPool --pool_blocks '{}' --dataset '{\"args\":{},\"_type\":\"RedditBinaryWrapper\"}' --num_epochs 3500 --add_layer 32 32 32 32 32 4 --output_layer_merge sum --wandb_name 'redditbin-gcn' --wandb_tags final gcn --seed
```
### DiffPool
```
--conv_type DenseGCNConv --dense_data --pooling_type DiffPool DiffPool DenseNoPool --pool_blocks '{\"num_output_nodes\":10}' '{\"num_output_nodes\":5}' '{}' --num_epochs 1500 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --dataset '{\"args\":{},\"_type\":\"RedditBinaryWrapper\"}' --pooling_loss_weight 1 --wandb_name 'redditbin-diffpool10_5' --wandb_tags final diffpool --seed
```
### ASAP
```
--conv_type GCNConv --sparse_data --pooling_type ASAP ASAP SparseNoPool --pool_blocks '{\"ratio_output_nodes\":0.5}' '{\"ratio_output_nodes\":0.5}' '{}' --num_epochs 1500 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --dataset '{\"args\":{},\"_type\":\"RedditBinaryWrapper\"}' --output_layer_merge sum --wandb_name 'redditbin-asap50%' --wandb_tags final asap --batch_size 32 --seed
```
## BBBP
### Ours
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"KMeans\",\"num_concepts\":15,\"global_clusters\":false,\"kmeans_threshold\":0.02}' '{\"cluster_alg\":\"KMeans\",\"num_concepts\":15,\"global_clusters\":false,\"kmeans_threshold\":0.02}' '{}' --num_epochs 5000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --dataset '{\"args\":{},\"_type\":\"BBBPAtomWrapper\"}' --output_layer_merge sum --wandb_name 'BBBP-kmeans15_15_2%' --wandb_tags final kmeans-merge --seed 
```
### GCN
```
--conv_type DenseGCNConv --dense_data --pooling_type DenseNoPool --pool_blocks '{}' --dataset '{\"args\":{},\"_type\":\"BBBPAtomWrapper\"}' --num_epochs 5000 --add_layer 32 32 32 32 32 4 --output_layer_merge sum --wandb_name 'BBBP-gcn' --wandb_tags final gcn --seed 
```
### DiffPool
```
--conv_type DenseGCNConv --dense_data --pooling_type DiffPool DiffPool DenseNoPool --pool_blocks '{\"num_output_nodes\":10}' '{\"num_output_nodes\":5}' '{}' --num_epochs 5000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --dataset '{\"args\":{},\"_type\":\"BBBPAtomWrapper\"}' --pooling_loss_weight 1 --wandb_name 'BBBP-diffpool10_5' --wandb_tags final diffpool --seed
```
### ASAP
```
--conv_type GCNConv --sparse_data --pooling_type ASAP ASAP SparseNoPool --pool_blocks '{\"ratio_output_nodes\":0.5}' '{\"ratio_output_nodes\":0.5}' '{}' --num_epochs 5000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --dataset '{\"args\":{},\"_type\":\"BBBPAtomWrapper\"}' --output_layer_merge sum --wandb_name 'BBBP-asap50%' --wandb_tags final asap --seed
```
### Hyperplane
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"KMeans\",\"num_concepts\":15,\"global_clusters\":false,\"kmeans_threshold\":0.02,\"num_mc_samples\":10,\"perturbation\":{\"_type\":\"GaussianPerturbation\",\"args\":{\"std\":0.1}}}' '{\"cluster_alg\":\"KMeans\",\"num_concepts\":15,\"global_clusters\":false,\"kmeans_threshold\":0.02,\"num_mc_samples\":10,\"perturbation\":{\"_type\":\"GaussianPerturbation\",\"args\":{\"std\":0.1}}}' '{}' --num_epochs 5000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --dataset '{\"args\":{},\"_type\":\"BBBPAtomWrapper\"}' --blackbox_transparency 0.3 --output_layer_merge sum --wandb_name 'BBBP-hyperplane-kmeans15_15_2%' --wandb_tags final hyperplane-kmeans-merge --seed
```
### Sequential
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"SequentialKMeansMeanShift\",\"global_clusters\":false,\"num_sketches\":60,\"mean_shift_range\":0.02,\"min_samples_per_sketch\":0.001,\"cluster_decay_factor\":0.9,\"rescale_clusters_decay\":0.8}' '{\"cluster_alg\":\"SequentialKMeansMeanShift\",\"global_clusters\":false,\"num_sketches\":60,\"mean_shift_range\":0.02,\"min_samples_per_sketch\":0.001,\"cluster_decay_factor\":0.9,\"rescale_clusters_decay\":0.8}' '{}' --dataset '{\"args\":{},\"_type\":\"BBBPAtomWrapper\"}' --num_epochs 5000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --wandb_name 'mutag-sequentialkmeans' --wandb_tags final sequentialkmeans-merge --seed 
```
### Global
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"KMeans\",\"num_concepts\":15,\"global_clusters\":true,\"kmeans_threshold\":0.0}' '{\"cluster_alg\":\"KMeans\",\"num_concepts\":15,\"global_clusters\":true,\"kmeans_threshold\":0.0}' '{}' --num_epochs 5000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --dataset '{\"args\":{},\"_type\":\"BBBPAtomWrapper\"}' --output_layer_merge sum --wandb_name 'BBBP-kmeans15_15_global' --wandb_tags final kmeans-global --seed 
```
## Expressiveness
### Ours
```
--conv_type DenseGCNConv --dense_data --pooling_type MonteCarlo MonteCarlo DenseNoPool --pool_blocks '{\"cluster_alg\":\"KMeans\",\"num_concepts\":10,\"global_clusters\":false,\"kmeans_threshold\":0.02}' '{\"cluster_alg\":\"KMeans\",\"num_concepts\":15,\"global_clusters\":false,\"kmeans_threshold\":0.02}' '{}' --num_epochs 100 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --wandb_name 'express-kmeans10_15_2%' --wandb_tags final kmeans-merge --dataset '{\"args\":{},\"_type\":\"ExpressivityWrapper\"}' --seed
```
### GCN
```
--conv_type DenseGCNConv --dense_data --pooling_type DenseNoPool --pool_blocks '{}' --num_epochs 1000 --add_layer 32 32 32 32 32 4 --output_layer_merge sum --wandb_name 'express-gcn' --wandb_tags final gcn --dataset '{\"args\":{},\"_type\":\"ExpressivityWrapper\"}' --seed 
```
### DiffPool
```
--conv_type DenseGCNConv --dense_data --pooling_type DiffPool DiffPool DenseNoPool --pool_blocks '{\"num_output_nodes\":10}' '{\"num_output_nodes\":5}' '{}' --num_epochs 1000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --output_layer_merge sum --dataset '{\"args\":{},\"_type\":\"ExpressivityWrapper\"}' --pooling_loss_weight 1 --wandb_name 'express-diffpool10_5' --wandb_tags final diffpool --seed
```
### ASAP
```
--conv_type GCNConv --sparse_data --pooling_type ASAP ASAP SparseNoPool --pool_blocks '{\"ratio_output_nodes\":0.5}' '{\"ratio_output_nodes\":0.5}' '{}' --num_epochs 1000 --add_layer 32 32 --add_layer 32 32 32 --add_layer 32 32 4 --dataset '{\"args\":{},\"_type\":\"ExpressivityWrapper\"}' --output_layer_merge sum --wandb_name 'express-asap50%' --wandb_tags final asap --seed
```