"""
DDIT: A D_ata-D_riven I_nformation T_heory framework.

Vincent Ragusa 2019-2025
"""
import polars as pl  # pip3 install --user polars


def make_expression(shared_parts:list[str],condition:list[str]) -> str:
    """Makes a valid entropy expression from the given components.

    Args:
        shared_parts (list[str]): The variables on the shared side of the condition (one or more)
        condition (list[str]): The variables on the joint side of the condition (zero or more)

    Returns:
        str: The corresponding entropy expression
    """
    return ":".join(shared_parts)+ ("|" if condition else "") + "&".join(condition)


class DDIT:
    """DDIT class contains all functionality for working with data
        and computing information theoretic quantities.
    """
    def __init__(self, verbose:bool=False) -> None:
        self.verbose:bool = verbose
        self.df:pl.DataFrame = None


    def read_csv(self,path_to_csv:str) -> None:
        """loads data from the specified csv file.

        Args:
            path_to_csv (str): path to csv file 
        """
        self.df = pl.read_csv(path_to_csv)

        if self.verbose:
            print(f"Reading data from {path_to_csv}")
            print(self.df)


    def add_column(self,new_data:dict[str,list]) -> None:
        """Adds data to DDIT's dataFrame. If the dataframe has not already been
        initialized, this function will initialize it.

        Args:
            new_data (dict[str,list]): A python dictionary containing named data.
            the length of the data fields must match any existing data fields in
            DDIT's dataframe.
        """
        temp_df = pl.DataFrame(new_data,strict=False)

        if self.df is None:
            self.df = temp_df
            if self.verbose:
                print("Initializing dataframe.")
                print(self.df)
        else:
            self.df = pl.concat([self.df,temp_df],how="horizontal")
            if self.verbose:
                print("Extending dataframe.")
                print(self.df)


    def entropy(self,column_name:str) -> pl.DataFrame:
        """Computes the Shannon entropy of the column specified.

        Args:
            column_name (str): column identifier 

        Returns:
            pl.DataFrame: a dataframe containing the Shannon Entropy
        """
        if self.verbose:
            print(f"Computing the entropy of {column_name}")
        alias = f"H({column_name})"
        select = pl.col(column_name) .cast(pl.Utf8) .unique_counts() .entropy(base=2) .alias(alias)
        return self.df.select(select)


    def join(self,column_names:list[str]) -> None:
        """Joins together two columns so their joint entropy can be computed at a later time.
        The joint column is added to the set of existing columns.

        Args:
            column_names (list[str]): list of column names to join
        """
        column_names.sort()
        if self.verbose:
            print(f"Joining columns {column_names}")
        alias = "&".join(column_names)
        with_columns = pl.struct(column_names) .alias(alias)
        self.df = self.df.with_columns(with_columns)


    def evaluate_expression(self,entropy_expression:str) -> pl.DataFrame:
        """Computes the value of arbitrary entropy expressions in 'normal form'.
        Expressions are decomposed using the following identities:\n
        A:B:C|D = A:B|D - A:B|C&D,\n
        A:B|C = A|C - A|B&C,\n
        A|B = A&B - B.\n
        Together, these expressions allow for a recursive decomposition of any
        formula into the sum of joint entropies.

        Args:
            entropy_expression (str): An entropy expression in normal form.
            Normal form is 'X: ... :Y|Z', where X, Y, and Z can be single or joint
            random variables and there can be arbitrarily many 'shared' variables.
            No single variable can appear more than once. Use joint variable aliases
            to prevent multiple appearences of a variable if necessary. It is never
            necessary.

        Returns:
            pl.DataFrame: DataFrame containing the entropy of the given expression.
        """
        split_conditional = entropy_expression.split("|")
        shared_parts = sorted(split_conditional[0].split(":"))
        condition = sorted([var for chunk in split_conditional[1:] for var in chunk.split("&")])

        # print("Entered With", make_expression(shared_parts,condition))

        if len(shared_parts) > 1:
            #A:B:C|D = A:B|D - A:B|C&D
            #A:B|C = A|C - A|B&C
            shifted_var = shared_parts.pop().split("&")
            lhs = make_expression(shared_parts,condition)
            # print("LHS", lhs)
            condition.extend(shifted_var)
            condition.sort()
            rhs = make_expression(shared_parts,condition)
            # print("RHS", rhs)
            joint_df = pl.concat([self.evaluate_expression(lhs), self.evaluate_expression(rhs)],
                                 how="horizontal")
            alias = f"H({entropy_expression})"
            return joint_df.select((pl.col(f"H({lhs})") - pl.col(f"H({rhs})")).alias(alias))

        if condition:
            #A|B = A&B - B
            lhs = make_expression(["&".join(sorted(condition+shared_parts[0].split("&")))],[])
            rhs = make_expression(["&".join(condition)],[])
            # print("LHS", lhs)
            # print("RHS", rhs)
            joint_df = pl.concat([self.evaluate_expression(lhs), self.evaluate_expression(rhs)],
                                 how="horizontal")
            alias = f"H({entropy_expression})"
            return joint_df.select((pl.col(f"H({lhs})") - pl.col(f"H({rhs})")).alias(alias))

        variables = shared_parts[0].split("&")

        if len(variables) > 1:
            # Evaluate joint random variable
            sorted_expression = "&".join(sorted(variables))
            if sorted_expression not in self.df:
                self.join(variables)
            return self.entropy(sorted_expression)

        return self.entropy(variables[0])





if __name__ == "__main__":
    ddit = DDIT(verbose=False)
    # ddit.read_csv("./xor_data.csv")

    my_data = {"X":[0,0,1,1],
               "Y":['a','b','a','b']}
    ddit.add_column(my_data)
    ddit.add_column({"Z":["a",0,0,"a"]})

    ee = ddit.evaluate_expression("X:Y|Z")
    print(ee)
    print(ddit.df)
