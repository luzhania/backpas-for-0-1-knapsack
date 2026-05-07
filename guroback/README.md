# GuroBack  
GuroBack is a native backbone extractor for Pseudo-Boolean Optimization based on Gurobi.  

## Requirements  
GuroBack depends on the Gurobi C++ API. To use GuroBack, you must:  
- Have Gurobi installed on your machine.  
- Possess a valid Gurobi license.  

GuroBack has been tested with Gurobi 10 and Gurobi 11.  

## Compilation  
Before compiling, update the `GUROBI_INSTALL_DIR` parameter in the `Makefile` to point to the directory where Gurobi is installed. This directory should contain the `include` and `lib` folders.  

Inside the `lib` folder, the following files must be present:  
- `libgurobiXXX.so` (e.g., `libgurobi110.so` for Gurobi 11.0)  
- `libgurobi_c++.a`  

To compile GuroBack, run:  

```sh
make guroback
```  

## Usage  
To run GuroBack, use the following command:  

```sh
./guroback [parameters] <instanceFile> <backboneFile>
```  

Any additional parameters are passed directly to Gurobi. For example:  

```sh
./guroback Threads=4 TimeLimit=3600 FeasibilityTol=1e-9 OptimalityTol=1e-9 IntFeasTol=1e-9 <instanceFile> <backboneFile>
```  

This command extracts the backbone of `<instanceFile>`, using:  
- 4 threads  
- A 1-hour time limit (applied to each Gurobi call, not GuroBack itself)  
- Tolerances set to `1e-9`  

### Specifying the Optimum  
If the optimal value of an instance is known, GuroBack can use this information to speed up the backbone extraction process.  

GuroBack provides a special parameter that is **not** passed to Gurobi:  

- **`GurobackOptimum=<value>`**  
  - If specified, GuroBack assumes that the optimal value of the instance is `<value>`.  
  - The backbone is computed considering only feasible solutions with this specific value.  
