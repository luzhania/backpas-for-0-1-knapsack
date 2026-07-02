# BackPaS for 0-1 Knapsack: An Experimental Evaluation

This repository contains the source code, experimental environment, and datasets for the paper **"BackPaS for 0-1 Knapsack: An experimental evaluation"**.

The 0-1 Knapsack Problem is often considered one of the more tractable NP-hard problems; however, "correlated" instances remain a formidable challenge for modern exact solvers. This project evaluates the application of the Backbone-based Predict and Search (BackPaS) framework—a method originally specialized for Pseudo-Boolean Optimization—to tackle the highly difficult *bounded strongly correlated* instances of the 0-1 Knapsack Problem. 

By using Graph Neural Networks (GNNs) to predict backbone variables (variable-value pairs fixed across all optimal solutions), we construct an adaptive trust region that dynamically restricts the solver's search space, accelerating the exact solving process.

## Key Libraries & Environment

The most critical libraries utilized in this project are:
- **`gurobi` (v12.0.3)**: For exact solving and iterative backbone extraction.
- **`optuna` (v4.8.0)**: For Bayesian hyperparameter optimization of the trust region parameters.

You can use the provided `environment.yml` file to fully recreate the environment with all the required dependencies.

## Credits and Origin

The algorithmic core of this project is based on the original repositories by Bryan Alvarado Ulloa:
- [backpas](https://github.com/bryan-alvarado-ulloa/backpas): A predict-and-search framework that integrates Machine Learning into the solution process of optimization problems using Pseudo-Boolean Optimization (PBO) instances with backbone training.
- [guroback](https://github.com/bryan-alvarado-ulloa/guroback): A native backbone extractor for Pseudo-Boolean Optimization based on Gurobi.

These repositories were adapted and modified in their input pipeline to support the new combinatorial optimization instances (specifically 0-1 Knapsack problems converted to PBO format) proposed in our work.

## Repository Structure

The project is organized into the following directories and scripts:

- **[`backpas/`](backpas)**: Contains the core Python implementation of the Backbone-based Predict-and-Search (BackPaS) framework. It handles GNN model training, dataset formatting, and trust region evaluations.
- **[`guroback/`](guroback)**: Contains the C++ code for GuroBack, the native backbone extractor utilizing Gurobi.
- **[`knapsack/`](knapsack)**: Contains the C++ knapsack instance generator (`gen2`) and helper scripts to convert knapsack instances into `.opb` (Pseudo-Boolean Optimization) format.

## Datasets

The datasets corresponding to each of the evaluation scenarios are located under the **[`backpas/dataset/`](backpas/dataset)** directory. Each scenario directory contains the corresponding Pseudo-Boolean Optimization `.opb` instances in its `instance/` subfolder.

---

## Getting Started

### 1. Prerequisites

- **C++ Compiler**: A compiler supporting C++11 (e.g., `g++` or `clang++`).
- **Gurobi Optimizer**: A valid Gurobi installation and license.
- **Python Environment**: You can use the provided `environment.yml` file to recreate the exact environment required to run the scripts.

### 2. Compiling GuroBack

Configure your Gurobi installation path by editing the `GUROBI_INSTALL_DIR` parameter in [`guroback/Makefile`](guroback/Makefile), then run:

```sh
cd guroback
make guroback
cd ..
```

### 3. Generating Knapsack Instances

The `knapsack` folder provides tools to generate 0-1 knapsack instances and convert them to the OPB format:

1. Compile the generator and generate the raw `.in` instances:
   ```sh
   cd knapsack/src
   ./generate_instances.sh
   cd ../..
   ```
2. Batch-convert the `.in` files to `.opb` format:
   ```sh
   cd knapsack/src
   ./convert_all_to_opb.sh
   cd ../..
   ```
   *Note: This automatically places the converted `.opb` files in the `backpas/dataset/<SCENARIO_FOLDER>/instance/` directory.*

### 4. Execution Workflow

With the `.opb` instances ready, run the following pipeline scripts in order:

1. **Pipeline Execution (Extraction & Training)**:
   ```sh
   ./pipeline.sh
   ```
   This extracts backbones, builds the graph dataset, divides the data into train/validation/test partitions, and trains the Graph Convolutional Network.

2. **Validation**:
   ```sh
   ./validation.sh
   ```
   This validates the trust region parameters using the trained model to find the best configuration (via Optuna).

3. **Testing**:
   ```sh
   ./test.sh
   ```
   This builds the final trust regions and runs Gurobi evaluations on the test set instances.
