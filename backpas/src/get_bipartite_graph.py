import pyscipopt as scp
import torch
from constants import VARIABLES_GRAPH, LITERALS_GRAPH
def normalize_model(model,add_complement_constraints=False):
    """
    Normalize the SCIP model such that all coefficients in constraints are non-negative.
    This may involve duplicating binary variables and adjusting constraints accordingly.
    
    Args:
        model (pyscipopt.scip.Model): The SCIP model to be normalized.
    """
    var_map = {}  # Map to store original variables and their complements
    complement_constraints = {}
    used_complement_vars = set()  # Set to store complement variables that are used in constraints
    # Step 1: Duplicate binary variables with complements
    for var in model.getVars():
        if var.vtype() == "BINARY":  # Check if the variable is binary
            var_map[var.name] = var  # Store the original variable
            # Create a complement variable
            complement_var = model.addVar(name=f"{var.name}_complement", vtype="B")
            var_map[complement_var.name] = complement_var
            complement_constraints[complement_var.name] = (var + complement_var == 1)
    # Step 2: Update constraints
    for cons in model.getConss():
        vals_linear = model.getValsLinear(cons)  # Dictionary of variables and coefficients
        lhs = model.getLhs(cons)  # Left-hand side of the constraint
        rhs = model.getRhs(cons)  # Right-hand side of the constraint
        # Create new expressions for the updated constraint
        new_coefs = {}
        offset = 0
        for var_name, coef in vals_linear.items():

            if coef < 0:  # Handle negative coefficients
                complement_var = var_map[f"{var_name}_complement"]
                new_coefs[complement_var.name] = -coef
                offset -= coef
                used_complement_vars.add(complement_var.name)
            else:
                new_coefs[var_name] = coef
        
        # Remove the old constraint
        model.delCons(cons)

        # Add the new constraint with normalized coefficients
        lin_exp = sum(new_coefs.get(var_name) * var_map.get(var_name) for var_name in new_coefs.keys())
        if lhs!=-model.infinity():
            assert lhs+offset>=-model.infinity()
            model.addCons(
                lin_exp >= (lhs+offset)
            )
        if rhs!=model.infinity():
            assert rhs+offset<=model.infinity()
            model.addCons(
                lin_exp <= (rhs+offset)
            )
     # Step 3: Update the objective function
    obj_exp = model.getObjective()
    new_obj_exp = {}
    obj_offset = 0
    for term, coef in obj_exp.terms.items():
        assert len(term.vartuple) == 1 # verify that the term is a monomial (with only 1 variable)
        var_name = term.vartuple[0].name
        if coef < 0:
            complement_var = var_map[f"{var_name}_complement"]
            new_obj_exp[complement_var.name] = -coef
            obj_offset += coef
            used_complement_vars.add(complement_var.name)
        else:
            new_obj_exp[var_name] = coef
    obj_lin_exp = sum(new_obj_exp.get(var_name) * var_map.get(var_name) for var_name in new_obj_exp.keys())
    model.setObjective(
        obj_lin_exp,
        sense = model.getObjectiveSense(),
        clear = True
    )
    model.addObjoffset(obj_offset, solutions=False)
    if add_complement_constraints:
        for used_complement_var in used_complement_vars:
            model.addCons(complement_constraints[used_complement_var])
def normalize_features(features : str):
    maxs = torch.max(features, 0)[0] # maximums of each feature (the [0] is because max returns a tuple of (values, indices))
    mins = torch.min(features, 0)[0] # minimums of each feature (the [0] is because min returns a tuple of (values, indices))
    diff = maxs - mins # Range of each feature
    for ks in range(diff.shape[0]): # iterate over each feature
        if diff[ks] == 0: # If the feature is constant, its range is assumed to be 1
            diff[ks] = 1
    # normalized so that features are between 0 and 1
    return torch.clamp((features - mins)/ diff, 1e-5, 1) # clamp to be between 1e-5 and 1

def get_bipartite_graph_literals(ins_name : str):    
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)
    normalize_model(m)
    
    
    cons = m.getConss() # list of constraints
    # Remove constraints without coefficients from the list
    new_cons = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            continue
        new_cons.append(c)
    cons = new_cons
    ncons = len(cons) # Number of non-empty constraints
    # Sort constraints according to the number of variables they have (ties broken by constraint name)
    cons_map = [[x, len(m.getValsLinear(x))] for x in cons]
    cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
    cons = [x[0] for x in cons_map]

    l_map = {} # dictionary of type {l_name: l_index}
    l_nodes = []
    #literal embedding
    # [position] name: meaning
    # [0] obj: normalized coefficient of literal in the objective function
    # [1] l_coeff: average coefficient of the literal in all constraints
    # [2] Nl_coeff: degree of literal node in the bipartite representation (number of constraints the LITERAL appears in)
    # [3] max_coeff: maximum value among all coefficients of the literal
    # [4] min_coeff: minimum value among all coefficients of the literal
    # [5] Nv_coeff: degree of variable node in the bipartite representation (number of constraints the VARIABLE appears in)
    mvars = m.getVars() # list of variables
    mvars.sort(key=lambda v: v.name) # sort variables by name
    for i in range(len(mvars)): # iterate over each variable
        l_nodes.append([0,0,0,0,m.infinity(),0])
    
    l_map = {} # dictionary of type {variableName: index}
    indx = 0
    for v in mvars:
        if v.name.endswith("_complement"):
            continue
        l_map[v.name] = 2*indx
        l_map[v.name+"_complement"] = 2*indx + 1
        indx+=1
    c_nodes = []
    #constraint embedding
    # [position] name: meaning
    # [0] c_coeff: average of all coefficients in the constraint
    # [1] Nc_coeff: degree of constraint nodes in the bipartite representation
    # [2] rhs: right-hand-side value of the constraint
    # [3] sense: the sense of the constraint
    indices_spr = [[], []] # Edge coordinates [constraintIndexes, variableIndexes]
    values_spr = [] # Weight for the adjacency matrix

    for cind, c in enumerate(cons): # iterate over constraints
        coeff = m.getValsLinear(c) # dictionary of coefficients {variableName: coefficient}
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        assert rhs!=m.infinity() or lhs!=-m.infinity() # at least one of the two must be finite
        #this assumes that the constraint is of the form lhs <= expr <= rhs with only one of lhs or rhs being finite or both being equal
        if lhs == rhs:
            sense = 0 # exp = rhs
        elif lhs != -m.infinity():
            sense = 1 # exp >= lhs
            rhs = lhs
        elif rhs != m.infinity():
            sense = -1 # exp <= rhs

        summation = 0
        for lit in coeff: # Iterate over variable names
            if coeff[lit] != 0: # If the coefficient is not 0, add the edge connecting the constraint with the literal
                l_indx = l_map[lit] # get the index of the literal
                indices_spr[0].append(cind) # index of the constraint
                indices_spr[1].append(l_indx) # index of the variable
                values_spr.append(coeff[lit]) # coefficient value (edge weight)
            
                l_nodes[l_indx][1] += coeff[lit] / ncons # Add normalized coefficient (according to constraint) to variable embedding
                l_nodes[l_indx][2] += 1 # Add 1 to the count of constraints in which the literal participates
                
                l_nodes[l_indx][3] = max(l_nodes[l_indx][3], coeff[lit]) # Update maximum coefficient of the variable
                l_nodes[l_indx][4] = min(l_nodes[l_indx][4], coeff[lit]) # Update minimum coefficient of the variable
                l_nodes[l_indx][5] += 1 # Add 1 to the count of constraints in which the variable participates
                neg_lit = lit+"_complement" if not lit.endswith("_complement") else lit[:-11]
                l_nodes[l_map[neg_lit]][5] = l_nodes[l_indx][5] # Update the count also in its negation

                summation += coeff[lit] # Accumulate all coefficients of the constraint
        llc = max(len(coeff), 1) # number of coefficients in the constraint (assumes at least 1)
        c_nodes.append([summation / llc, llc, rhs, sense]) # add the constraint embeddings
    obj = m.getObjective()
    is_optimization = len(obj.terms)!=0
    if is_optimization:
        obj_node = [0, 0, 0, 0] # node (constraint) representing the objective function
        for e in obj: # terms in the objective function
            vnm = e.vartuple[0].name # variable name
            v = obj[e] # variable coefficient
            #v_indx = v_map[vnm] # variable index
            if v != 0: # If coefficient is not 0, add edge connecting variable to constraint (objective function)
                if v>0:
                    lit = vnm
                else:
                    lit = vnm+"_complement"
                l_indx = l_map[lit] # get literal index
                indices_spr[0].append(ncons) # Index of constraint (objective function)
                indices_spr[1].append(l_indx) # Literal index
                values_spr.append(v)
                l_nodes[l_indx][0] = v # add the coefficient of the variable in the objective function to the variable embedding

                obj_node[0] += v # accumulate coefficients of all variables
                obj_node[1] += 1 # counts how many coefficients there are (how many variables)
        
        obj_node[0] /= obj_node[1] # calculate average of coefficients in constraint (objective function)
        c_nodes.append(obj_node)
        ncons+=1
    
    l_nodes = torch.as_tensor(l_nodes, dtype=torch.float32)#.to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32)#.to(device)
    l_nodes = normalize_features(l_nodes)
    c_nodes = normalize_features(c_nodes)

    A = torch.sparse_coo_tensor(indices_spr, values_spr, (ncons, len(mvars)*2))
    return A, l_map, l_nodes, c_nodes

def get_bipartite_graph_variables(ins_name):
    # Same function used by PaS and ConPaS. Code extracted from https://github.com/sribdcn/Predict-and-Search_MILP_method/blob/main/helper.py
    # The only difference is that here the tensors are not moved to a specific device.
    epsilon = 1e-6

    # vars:  [obj coeff, norm_coeff, degree, Bin?]
    m = scp.Model()
    m.hideOutput(True)
    m.readProblem(ins_name)

    ncons = m.getNConss()
    nvars = m.getNVars()

    mvars = m.getVars()
    mvars.sort(key=lambda v: v.name)


    v_nodes = []

    b_vars = []

    ori_start = 6
    emb_num = 15

    for i in range(len(mvars)):
        tp = [0] * ori_start
        tp[3] = 0
        tp[4] = 1e+20
        # tp=[0,0,0,0,0]
        if mvars[i].vtype() == 'BINARY':
            tp[ori_start - 1] = 1
            b_vars.append(i)

        v_nodes.append(tp)
    v_map = {}

    for indx, v in enumerate(mvars):
        v_map[v.name] = indx

    obj = m.getObjective()
    obj_cons = [0] * (nvars + 2)
    indices_spr = [[], []]
    values_spr = []
    obj_node = [0, 0, 0, 0]
    for e in obj:
        vnm = e.vartuple[0].name
        v = obj[e]
        v_indx = v_map[vnm]
        obj_cons[v_indx] = v
        if v != 0:
            indices_spr[0].append(0)
            indices_spr[1].append(v_indx)
            # values_spr.append(v)
            values_spr.append(1)
        v_nodes[v_indx][0] = v

        # print(v_indx,float(nvars),v_indx/float(nvars),v_nodes[v_indx][ori_start:ori_start+emb_num])

        obj_node[0] += v
        obj_node[1] += 1
    obj_node[0] /= obj_node[1]
    # quit()

    cons = m.getConss()
    new_cons = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        if len(coeff) == 0:
            # print(coeff,c)
            continue
        new_cons.append(c)
    cons = new_cons
    ncons = len(cons)
    cons_map = [[x, len(m.getValsLinear(x))] for x in cons]

    cons_map = sorted(cons_map, key=lambda x: [x[1], str(x[0])])
    cons = [x[0] for x in cons_map]

    lcons = ncons
    c_nodes = []
    for cind, c in enumerate(cons):
        coeff = m.getValsLinear(c)
        rhs = m.getRhs(c)
        lhs = m.getLhs(c)
        # A[cind][-2]=rhs
        sense = 0

        if rhs == lhs:
            sense = 2
        elif rhs >= 1e+20:
            sense = 1
            rhs = lhs

        summation = 0
        for k in coeff:
            v_indx = v_map[k]
            # A[cind][v_indx]=1
            # A[cind][-1]+=1
            if coeff[k] != 0:
                indices_spr[0].append(cind)
                indices_spr[1].append(v_indx)
                values_spr.append(1)
            v_nodes[v_indx][2] += 1
            v_nodes[v_indx][1] += coeff[k] / lcons
            v_nodes[v_indx][3] = max(v_nodes[v_indx][3], coeff[k])
            v_nodes[v_indx][4] = min(v_nodes[v_indx][4], coeff[k])
            # v_nodes[v_indx][3]+=cind*coeff[k]
            summation += coeff[k]
        llc = max(len(coeff), 1)
        c_nodes.append([summation / llc, llc, rhs, sense])
    c_nodes.append(obj_node)
    v_nodes = torch.as_tensor(v_nodes, dtype=torch.float32)#.to(device)
    c_nodes = torch.as_tensor(c_nodes, dtype=torch.float32)#.to(device)
    b_vars = torch.as_tensor(b_vars, dtype=torch.int32)#.to(device)

    A = torch.sparse_coo_tensor(indices_spr, values_spr, (ncons + 1, nvars))#.to(device)
    clip_max = [20000, 1, torch.max(v_nodes, 0)[0][2].item()]
    clip_min = [0, -1, 0]

    v_nodes[:, 0] = torch.clamp(v_nodes[:, 0], clip_min[0], clip_max[0])

    maxs = torch.max(v_nodes, 0)[0]
    mins = torch.min(v_nodes, 0)[0]
    diff = maxs - mins
    for ks in range(diff.shape[0]):
        if diff[ks] == 0:
            diff[ks] = 1
    v_nodes = v_nodes - mins
    v_nodes = v_nodes / diff
    v_nodes = torch.clamp(v_nodes, 1e-5, 1)
    # v_nodes=position_get_ordered(v_nodes)
    # v_nodes=position_get_ordered_flt(v_nodes)

    maxs = torch.max(c_nodes, 0)[0]
    mins = torch.min(c_nodes, 0)[0]
    diff = maxs - mins
    c_nodes = c_nodes - mins
    c_nodes = c_nodes / diff
    c_nodes = torch.clamp(c_nodes, 1e-5, 1)

    return A, v_map, v_nodes, c_nodes, b_vars

def transform_l_map_to_v_map(l_map:dict):
    v_map = {key: idx // 2 for key, idx in l_map.items() if idx % 2 == 0}
    for key, idx in l_map.items():
        if idx % 2 == 1:
            assert key[:-len("_complement")] in v_map, f"Variable {key} not found in v_map"
            assert v_map[key[:-len("_complement")]] == idx // 2, f"Variable {key} not found in v_map with the same index" 
    return v_map
def get_standard_bipartite_graph(instance_filepath:str, graph_type:str):
    if graph_type == VARIABLES_GRAPH:
        A, v_map, v_nodes, c_nodes, b_vars = get_bipartite_graph_variables(instance_filepath)
        bipartite_graph = (A, v_map, v_nodes, c_nodes)
    elif graph_type == LITERALS_GRAPH:
        A, l_map, l_nodes, c_nodes = get_bipartite_graph_literals(instance_filepath)
        v_map = transform_l_map_to_v_map(l_map)
        bipartite_graph = (A, v_map, l_nodes, c_nodes)
    else:
        raise ValueError(f"Unknown graph type requested: {graph_type}")
    return bipartite_graph