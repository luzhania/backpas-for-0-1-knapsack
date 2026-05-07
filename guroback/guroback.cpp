#include "gurobi_c++.h"
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_set>
#include <cmath>
#include <chrono>

using namespace std;

class Guroback
{
public:
     // Constructor
     Guroback(string instanceFile, string backboneFile, GRBEnv &grbEnv);
     // Constructor
     Guroback(string instanceFile, string backboneFile, GRBEnv &grbEnv, bool computeOptimum, double optimum);
     // Public methods
     void solve();

private:
     // Private attributes
     GRBModel model;
     vector<GRBVar> vars;
     vector<bool> values;
     unordered_set<int> candidates;
     unordered_set<int> backbones;
     ofstream backboneLog;
     GRBConstr constraint;
     double optimum;
     bool computeOptimum;
     bool modelIsOptimization;
     // Private helper methods
     bool flipAndSolve(GRBVar var, bool value);
     bool flipAndSolve(vector<int> chunk);
     inline bool toBool(double value);
     inline bool is_optimal();
     inline bool is_finished();
     inline void optimize_current();
     inline void fixObjectiveFunction(double optimum);
     inline void checkModelType();
};

// Constructor implementation
Guroback::Guroback(string instanceFile, string backboneFile, GRBEnv &grbEnv, bool computeOptimum, double optimum)
    : model(grbEnv, instanceFile), backboneLog(backboneFile)
{
     GRBVar *allVars = model.getVars();
     int numVars = model.get(GRB_IntAttr_NumVars);
     for (int i = 0; i < numVars; ++i)
     {
          if (allVars[i].get(GRB_CharAttr_VType) == GRB_BINARY)
          {
               vars.push_back(allVars[i]);
               candidates.insert(vars.size() - 1);
          }
     }
     delete[] allVars;
     this->computeOptimum = computeOptimum;
     this->optimum = optimum;
     checkModelType();
}

Guroback::Guroback(string instanceFile, string backboneFile, GRBEnv &grbEnv)
    : Guroback(instanceFile, backboneFile, grbEnv, true, 0){

    }

// Solve method implementation
void Guroback::solve()
{
     if(modelIsOptimization){
          cout << "Model is an optimization problem" << endl;
     } else{
          cout << "Model is a feasibility problem" << endl;
     }
     if(!computeOptimum){
          if(modelIsOptimization){
               cout << "Using optimum specified as parameter: " << optimum << endl;
               fixObjectiveFunction(optimum);
          } else {
               cout << "Model is a feasibility problem, ignoring optimum parameter" << endl;
          }
          
     }
     cout << "Solving the model" << endl;
     optimize_current();
     if (is_optimal())
     {
          if(modelIsOptimization){
               if(computeOptimum){
                    optimum = model.get(GRB_DoubleAttr_ObjVal);
                    cout << "Optimum is: " << optimum << endl;
                    fixObjectiveFunction(optimum);
               }else{
                    cout << "Optimum used leads to feasible solution" << endl;
               }
          } else {
               cout << "Instance is feasible" << endl;
          }
     }
     else
     {
          cout << "Instance is infeasible or unbounded" << endl;
          return;
     }

     cout << "Get solution values" << endl;
     for (size_t i = 0; i < vars.size(); i++)
     {
          values.push_back(toBool(vars[i].get(GRB_DoubleAttr_X)));
     }

     model.set(GRB_IntParam_MIPFocus, 1); // Focus on finding feasible solutions

     size_t chunk_size = 1;
     while (!candidates.empty())
     {
          vector<int> chunk;
          for (int i : candidates)
          {
               chunk.push_back(i);
               if (chunk.size() == chunk_size)
                    break;
          }

          cout << "Candidates=" << candidates.size() << " chunk_size=" << chunk_size << endl;
          auto result = flipAndSolve(chunk);

          if (!is_finished())
          {
               cout << "\tInterrupted" << endl;
               return;
          }

          if (result)
          {
               cout << "\tOptimum" << endl;
               for (auto it = candidates.begin(); it != candidates.end();)
               {
                    if (toBool(vars[*it].get(GRB_DoubleAttr_X)) != values[*it])
                    {
                         it = candidates.erase(it);
                    }
                    else
                    {
                         it++;
                    }
               }
               chunk_size = 1;
          }
          else
          {
               cout << "\tNOT Optimum" << endl;
               for (int i : chunk)
               {
                    candidates.erase(i);
                    backbones.insert(i);
                    //backboneLog << (values[i] ? "b " : "b -") << vars[i].get(GRB_StringAttr_VarName) << endl;
                    backboneLog << (values[i] ? "b " : "b -") << vars[i].get(GRB_StringAttr_VarName).substr(1) << endl;
               }
               chunk_size = min(2 * chunk_size, candidates.size());
          }
     }

     cout << "Backbone extraction complete" << endl;
     backboneLog << "b 0" << endl;
}

inline void Guroback::fixObjectiveFunction(double optimum){
     model.addConstr(model.getObjective() == optimum, "fix_objective");
     GRBLinExpr zeroObj = 0;
     model.setObjective(zeroObj, GRB_MINIMIZE);  // Remove the objective
}

bool Guroback::flipAndSolve(GRBVar var, bool value)
{
     var.set(GRB_DoubleAttr_UB, !value);
     var.set(GRB_DoubleAttr_LB, !value);
     optimize_current();

     if (!is_optimal() && is_finished())
     {
          var.set(GRB_DoubleAttr_UB, value);
          var.set(GRB_DoubleAttr_LB, value);
          return false;
     }
     else
     {
          var.set(GRB_DoubleAttr_UB, 1);
          var.set(GRB_DoubleAttr_LB, 0);
          return true;
     }
}

bool Guroback::flipAndSolve(vector<int> chunk)
{
     if (chunk.size() == 1)
          return flipAndSolve(vars[chunk[0]], values[chunk[0]]);

     GRBLinExpr lhs = 0;
     for (int i : chunk)
     {
          lhs += values[i] ? (1 - vars[i]) : vars[i];
     }

     GRBConstr constraint = model.addConstr(lhs >= 1);
     optimize_current();

     bool isOptimum = is_optimal();
     if (!isOptimum && is_finished())
     {
          for (int i : chunk)
          {
               vars[i].set(GRB_DoubleAttr_UB, values[i]);
               vars[i].set(GRB_DoubleAttr_LB, values[i]);
          }
     }

     model.remove(constraint);
     return isOptimum;
}

inline bool Guroback::is_optimal()
{
     return model.get(GRB_IntAttr_Status) == GRB_OPTIMAL;
}

inline bool Guroback::is_finished()
{
     auto status = model.get(GRB_IntAttr_Status);
     return status == GRB_OPTIMAL || status == GRB_INFEASIBLE || status == GRB_UNBOUNDED;
}

inline bool Guroback::toBool(double value)
{
     return value > 0.5;
}

inline void Guroback::optimize_current()
{
     model.optimize();
     cout << "RUNTIME: " << model.get(GRB_DoubleAttr_Runtime) << "seconds" << endl;
}
inline void Guroback::checkModelType(){
     // Check if the model has an objective function
     modelIsOptimization = false;
     for (int i = 0; i < model.get(GRB_IntAttr_NumVars); i++) {
          GRBVar var = model.getVar(i);
          if (var.get(GRB_DoubleAttr_Obj) != 0.0) {
               modelIsOptimization = true;
               break;
          }
     }
}
// Main function
int main(int argc, char *argv[])
{
     if (argc < 3)
     {
          cerr << "Usage: executable [parameters] <instanceFile> <backboneFile>" << endl;
          return 1;
     }

     string instanceFile = argv[argc - 2];
     string backboneFile = argv[argc - 1];

     GRBEnv env = GRBEnv();
     env.set(GRB_IntParam_OutputFlag, 0);
     env.set(GRB_DoubleParam_MIPGap, 0.0);
     double optimum = 0;
     bool computeOptimum = true;

     for (int i = 1; i < argc - 2; i++)
     {
          string arg(argv[i]);
          size_t eqPos = arg.find('=');
          if (eqPos != string::npos)
          {
               string paramName = arg.substr(0, eqPos);
               string paramValue = arg.substr(eqPos + 1);
               if(paramName == "GurobackOptimum"){
                    computeOptimum = false;
                    optimum = stod(paramValue);
               }else{
                    env.set(paramName.c_str(), paramValue.c_str());
               }
          }
          else
          {
               cerr << "Invalid argument format: " << arg << ". Expected format: ParamName=Value." << endl;
               return 1;
          }
     }

     try
     {
          auto start = chrono::high_resolution_clock::now();
          Guroback guroback(instanceFile, backboneFile, env, computeOptimum, optimum);
          guroback.solve();
          auto stop = chrono::high_resolution_clock::now();
          auto duration = chrono::duration_cast<chrono::microseconds>(stop - start);
          cout << "TOTAL RUNTIME: " << duration.count() << " microseconds" << endl;
     }
     catch (GRBException &e)
     {
          cerr << "Error code = " << e.getErrorCode() << endl;
          cerr << e.getMessage() << endl;
          return 1;
     }
     catch (...)
     {
          cerr << "Unexpected error occurred!" << endl;
          return 1;
     }

     return 0;
}
