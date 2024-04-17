# A data-driven information theory framework.
# Allows the computation and composition of shannon entropy and mutual information.
# Data is stored and manipulated as columns/rows of variable states; entropy is calculated from collapsing these tables
# Alternate probability estimators are supported
# 
# Vincent Ragusa 2019-2024
# 


#TODO add a flag to column interface functions to have each entry in the dict a file instead to conserve RAM



from collections import Counter
from datetime import datetime
from random import seed, shuffle
from csv import reader

from numpy import array, dot, isclose, log2


class DDIT:

    def __init__(self, verbose:bool=False):
        self.raw_data = None
        self.labels = None
        self.__columns:dict[str:tuple[str]] = {}
        self.column_keys:set[str] = set()
        self.verbose:bool = verbose
        self.events_recorded:int = None


    def __get_column(self, key:str)->tuple[str]:
        return self.__columns[key]


    def __set_column_tuple(self, key:str, value_list:tuple[str])->None:
        self.__columns[key] = value_list
        self.column_keys.add(key)


    def remove_column(self,key:str)->None:
        if self.verbose: print("{} Deleting {} from memory ...".format(str(datetime.now()),key))
        del self.__columns[key]
        self.column_keys.remove(key)


    def __columns_contains(self, key:str)->bool:
        return key in self.column_keys


    def __columns_empty(self)->bool:
        return False if self.column_keys else True


    def load_csv(self, file_path:str, header:bool=False, auto_register:bool=False)->None:
        if self.verbose: print("{} Loading data from {} ...".format(str(datetime.now()),file_path))
        with open(file_path, 'r') as csv_file:
            rdr = reader(csv_file)
            if header:
                self.labels = next(rdr)
                self.raw_data = [tuple(row) for row in rdr]
                if auto_register:
                    for col, label in enumerate(self.labels):
                        self.register_column(label,col)
            else:
                self.raw_data = [row for row in rdr]
                if auto_register:
                    print("WARNING you must manually register columns if no header is provided. Skipping auto-register...")


    def register_column(self, key:str, colIndex:int)->None:
        if self.verbose: print("{} Registering {} as {} ...".format(str(datetime.now()), colIndex,key))

        new_column = tuple([row[colIndex] for row in self.raw_data])

        len_col = len(new_column)
        if self.events_recorded is None:
            self.events_recorded = len_col
        assert len_col == self.events_recorded # all columns must be the same length (rows)
        
        if self.__columns_contains(key):
            print("WARNING: column \"{}\" has already been registered. Overwriting...".format(key))
        self.__set_column_tuple(key,new_column)


    def register_column_tuple(self, key:str, col:tuple[str])->None:
        if self.verbose: print("{} Registering custom column as {} ...".format(str(datetime.now()),key))

        len_col = len(col)
        if self.events_recorded is None:
            self.events_recorded = len_col
        assert len_col == self.events_recorded # all columns must be the same length (rows)

        if self.__columns_contains(key):
            print("WARNING: column \"{}\" has already been registered. Overwriting...".format(key))

        self.__set_column_tuple(key,col)

    
    def print_columns(self)->None:
        if self.__columns_empty():
            print("No registered columns to print.")
        else:
            for key in self.column_keys:
                print(key, self.__get_column(key))
        print()

    
    def join_and_register(self, col1:str, col2:str, new_key:str=None)->None:
        if self.verbose: print("{} Creating joint distribution: {}&{} ...".format(str(datetime.now()), col1,col2))
        if not self.__columns_contains(col1):
            print("ERROR column \"{}\" is not registered!".format(col1))
            return
        if not self.__columns_contains(col2):
            print("ERROR column \"{}\" is not registered!".format(col2))
            return
        if new_key is not None:
            key = new_key
        else:
            key = col1 + "&" + col2
        new_col = list(zip(self.__get_column(col1),self.__get_column(col2)))
        self.register_column_tuple(key,new_col)


    def H(self, colName:str, columnData:list[str] = None)->float:
        if not self.__columns_contains(colName) and columnData is None:
            print("ERROR column \"{}\" is not registered!".format(colName))
            return
        
        if columnData is not None:
            if self.verbose: print(str(datetime.now()), "colName ignored in H() because columnData was provided.")
            events_data = columnData
        else:
            events_data = self.__get_column(colName)

        event_counts = Counter(events_data).values()
        p_vector = array([count/ self.events_recorded for count in event_counts])
        return - dot(p_vector, log2(p_vector)) + 0.0


    def I(self, col1:str, col2:str)->float:
        and_key = col1 + "&" + col2
        if not self.__columns_contains(and_key):
            if self.verbose: print("{} registering column \"{}\"...".format(str(datetime.now()), and_key))
            self.join_and_register(col1,col2)
        i = self.H(col1) + self.H(col2) - self.H(and_key)
        return i


    def __venn_gen_power_set(self, variables, current=[]):
        if variables:
            return self.__venn_gen_power_set(variables[1:],current=current) + self.__venn_gen_power_set(variables[1:], current=current + [variables[0]])
        return [current]


    def __venn_make_formula(self, subset, column_keys):
        if subset:
            subset.sort()
            remainder = sorted(list(set(column_keys) - set(subset)))

            S=""
            S = ":".join(subset)
            if remainder: S += "|"
            S += "&".join(remainder)
            return S
        return ""


    def recursively_solve_formula(self, formula:str)->float:
        if "|" in formula:
            # formula is a conditional
            halves = formula.split("|")
            if ":" in halves[0]:
                #formula is a conditional with shared entropy on the lhs
                shareds = halves[0].split(":")
                left_formula = ":".join(shareds[1:]) + "|" + halves[1]
                right_formula = ":".join(shareds[1:]) + "|" + "&".join(sorted(halves[1].split("&") + [shareds[0]]))
            else:
                #formula is a conditional of only joints
                left_formula = "&".join(sorted(halves[1].split("&") + halves[0].split("&")))
                right_formula = halves[1]
            return self.recursively_solve_formula(left_formula) - self.recursively_solve_formula(right_formula)
        elif ":" in formula:
            #formula is shared only; treat as special case of above case
            shareds = formula.split(":")
            left_formula = ":".join(shareds[1:])
            right_formula = ":".join(shareds[1:]) + "|" + shareds[0]
            return self.recursively_solve_formula(left_formula) - self.recursively_solve_formula(right_formula)
        else:
            # formula is only a joint; calculate from data
            variables = formula.split("&")
            if len(variables) == 1:
                return self.H(formula)
            else:
                jointData = tuple(zip(*[self.__get_column(v) for v in variables]))
                return self.H(formula, columnData=jointData)

        
    #TODO: return values
    def solve_venn_diagram(self, column_keys:list[str]=None)->None:
        if column_keys is None:
            column_keys = list(self.column_keys)
        if self.verbose: print("{} Generating power set of {}...".format(str(datetime.now()), column_keys))
        power_set = self.__venn_gen_power_set(column_keys)
        if self.verbose: print("{} Generating venn diagram...".format(str(datetime.now())))
        else: print("Generating venn diagram...")
        for i, subset in enumerate(power_set):
            if i > 0:
                formula = self.__venn_make_formula(subset, column_keys)
                entropy = self.recursively_solve_formula(formula)
                if self.verbose: print("{} {} {} {}".format(str(datetime.now()),i, formula, entropy))
                else: print("{} {} {}".format(i, formula, entropy))


    def greedy_chain_rule(self, X, B_list, alpha=None):
        print("Applying Chain Rule to {}...".format("{}:{}".format(X, "&".join(B_list))))
        chosen = []
        if alpha is None: alpha = len(B_list)
        while len(chosen) < alpha:
            if chosen:
                max_B = max(B_list, key= lambda B: self.recursively_solve_formula( "{}:{}|{}".format(X, B, "&".join(chosen) )) )
                f = "{}:{}|{}".format(X, max_B, "&".join(chosen))
                print(f, self.entropies[f])
            else:
                max_B = max(B_list, key= lambda B: self.recursively_solve_formula("{}:{}".format(X,B) ))
                f = "{}:{}".format(X,max_B)
                print(f, self.entropies[f])
            chosen.append(max_B)
            B_list.remove(max_B)


    def greedy_condition_adder(self, focalVar, OtherVar_list, numChains=None):
        #most entropy explainable is given by joint everything
        print("Finding Minimal explanatory set for {}...".format(focalVar))
        f = "{}|{}".format(focalVar,"&".join(OtherVar_list))
        target = self.recursively_solve_formula(f)
        chosen = []
        if numChains is None: numChains = len(OtherVar_list)
        time = 1
        while len(chosen) < numChains:
            if chosen:
                best_other = min(OtherVar_list, key= lambda other: self.recursively_solve_formula( "{}|{}".format(focalVar, "&".join(chosen+[other]) )) )
                f = "{}|{}".format(focalVar, "&".join(chosen+[best_other]))
                print("{} |{}".format(time,best_other), self.entropies[f])
            else:
                best_other = min(OtherVar_list, key= lambda other: self.recursively_solve_formula("{}|{}".format(focalVar,other) ))
                f = "{}|{}".format(focalVar,best_other)
                print("{} |{}".format(time,best_other), self.entropies[f])
            chosen.append(best_other)
            OtherVar_list.remove(best_other)
            time += 1
            if isclose(self.entropies[f],target):
                return chosen


    def greedy_node_removal(self, focalVar, OtherVar_list, numChains=None):
        chosen = []
        N = len(OtherVar_list)-1
        if numChains is None: numChains = len(OtherVar_list)
        time = 1
        while len(chosen) < numChains and len(chosen) < N:
            best_other = min(OtherVar_list, key= lambda other: self.recursively_solve_formula("{}:{}|{}".format(focalVar,other,"&".join([b for b in OtherVar_list if b != other]))))
            f = "{}:{}|{}".format(focalVar,best_other,"&".join([b for b in OtherVar_list if b != best_other]))
            if not isclose(self.entropies[f],0):
                print(len(OtherVar_list), OtherVar_list)
                return
            print("{} ΔI {}".format(time,best_other), self.entropies[f])
            
            chosen.append(best_other)
            OtherVar_list.remove(best_other)
            time += 1
        last_other = OtherVar_list[0]
        f = "{}:{}".format(focalVar,last_other)
        print("{} ΔI {}".format(time,last_other), self.recursively_solve_formula(f))


    def smallest_explanatory_set(self, focalVar:str, OtherVars:list[str], keepVars:list[str]=[], best=None, target=None):
        #first call init
        if best is None:
            #smallest fully explanatory set is everything
            best = OtherVars
            #most entropy explainable is given by joint everything
            f = "{}|{}".format(focalVar,"&".join(OtherVars))
            target = self.recursively_solve_formula(f)
            print("TARGET",target)

        #can we do better than the reported best, with the choices available?
        #if adding another variable would make us the same size as the best, there's no point looking further.
        if len(keepVars)+1 == len(best):
            return best
        
        #sort overvars to speed up tree search
        OtherVars.sort(key= lambda other: self.recursively_solve_formula( "{}|{}".format(focalVar, "&".join(keepVars+[other]) )) )

        #if len(otherVars) == 0, skip this loop and return best.
        for i in range(len(OtherVars)):
            #each loop evaluates one of the choices. The recursive call below always includes the choice.
            # thus, we check including and excluding each variable. We also eliminate repeated sets by keeping the choices ordered.
            # i.e., if X|AB is tested in one branch, X|BA will not be tested in another branch.
            choices = OtherVars[i:]
            #we first check if including every available choice is worse than the target.
            #if including everything is worse, we cannot do better than the reported best.
            f = "{}|{}".format(focalVar,"&".join(keepVars+choices))
            boundTest = self.recursively_solve_formula(f)
            if not isclose(boundTest,target):
                return best
            #we now test if including only our first choice reaches the target.
            f = "{}|{}".format(focalVar,"&".join(keepVars+[choices[0]]))
            entropy = self.recursively_solve_formula(f)
            # print(f,entropy) #DEBUG PRINT
            # if including our choice meets the target, (and we are smaller than the best,) return the new best.
            # note, including more variables cannot improve things further, and we cannot break ties in a meaningful way.
            # therefore, making a recursive call is pointless, and considering more iterations of this loop is wasteful.
            if isclose(entropy,target):# and len(keepVars)+1 < len(best):
                print("Rejoice, A new best!",keepVars+[choices[0]],"\n")
                return keepVars+[choices[0]]
            #if we have not met the target, we must include another variable. Unsure of which to include, we ask SES.
            # the variable we are considering is added to the keepVars so that SES will consider it as given.
            #the best set and target are passed to keep SES as up-to-date as possible
            if len(keepVars) + 2 < len(best):
                best = self.smallest_explanatory_set(focalVar,choices[1:],keepVars+[choices[0]],best,target)
        #after evaluating all choices, return the best.
        return best


    def __get_var_list(self,formula:str)->list[str]:
        if "|" in formula:
            parts = formula.split("|")
            return self.__get_var_list(parts[0]) + self.__get_var_list(parts[1])
        elif ":" in formula:
            return formula.split(":")
        elif "&" in formula:
            return formula.split("&")
        else:
            return [formula]

    def __tokenize_formula(self,formula:str)->list[str]:
        if "|" in formula:
            parts = formula.split("|")
            return self.__tokenize_formula(parts[0]) + ["|"] + self.__tokenize_formula(parts[1])
        elif ":" in formula:
            Vars = formula.split(":")
            returnList = []
            for var in Vars:
                returnList.append(var)
                returnList.append(":")
            return returnList[:-1] #remove last :
        elif "&" in formula:
            Vars = formula.split("&")
            returnList = []
            for var in Vars:
                returnList.append(var)
                returnList.append("&")
            return returnList[:-1] #remove last &
        else:
            return [formula]
        

    def __get_temp_formula(self,formula:str,repNum:int)->str:
        tokens = self.__tokenize_formula(formula)
        newFormula = ""
        for token in tokens:
            if token not in [":","|","&"]:
                newFormula += "PR{}_{}".format(repNum,token)
            else:
                newFormula += token
        return newFormula

    def solve_with_permutation_pvalue(self,formula:str,reps:int=100,rseed:int=None)->tuple[float,tuple[float,float]]:
        if rseed is not None: seed(rseed)
        vars = self.__get_var_list(formula)
        entropy = self.recursively_solve_formula(formula)
        #collect bootstrap distribution
        distribution = []
        for rep in range(reps):
            #create clone columns
            # newFormula = formula
            newFormula = self.__get_temp_formula(formula,rep)
            for var in vars:
                shuffledValues = list(self.__columns[var])
                shuffle(shuffledValues)
                self.register_column_tuple("PR{}_{}".format(rep,var),tuple(shuffledValues))
                # newFormula = newFormula.replace(var,"BS{}_{}".format(rep,var))
            #get entropy
            distribution.append(self.recursively_solve_formula(newFormula))
            #remove columns
            for key in list(self.column_keys):
                if key.startswith("PR{}_".format(rep)):
                    self.remove_column(key)
        #estimate p-value
        p_left = 0
        p_right = 0
        for value in distribution:
            if value >= entropy:
                p_right += 1
            if value <= entropy:
                p_left += 1
        p_left /= reps
        p_right /= reps

        return entropy, (p_left,p_right)



if __name__ == "__main__":

    # create an instance of the class
    ddit = DDIT(verbose=True)
    
    # auto register columns based on CSV headers 
    ddit.load_csv("xor_data.csv", header=True, auto_register=True)

    # display registered columns
    ddit.print_columns()

    # get the venn diagram of the system
    ddit.solve_venn_diagram(column_keys=["X","Y","Z"])

    # calculate an arbitrary entropy given in standard form
    entropy = ddit.recursively_solve_formula("X:Y|Z")
    print("The entropy of X:Y|Z is ", entropy)
