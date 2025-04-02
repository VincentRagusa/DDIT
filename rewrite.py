import polars as pl  # pip install --user polars








df = pl.DataFrame({"a": [1, 2, 3]})
print(df.select(pl.col("a").entropy(base=2)))


class DDIT:
    
    def __init__(self, verbose:bool=False) -> None:
        self.verbose:bool = verbose
        self.df:pl.DataFrame = None
        
    def read_csv(self,path_to_csv):
        self.df = pl.read_csv(path_to_csv)
        if self.verbose:
            print(f"Reading data from {path_to_csv}")
            print(self.df)
        
    def entropy(self,column_name):
        if self.verbose: print(f"Computing the entropy of {column_name}")
        return self.df.select(pl.col(column_name).unique_counts().entropy(base=2,normalize=True).alias(f"H({column_name})"))
    
    
if __name__ == "__main__":
    ddit = DDIT(verbose=True)
    
    ddit.read_csv("./xor_data.csv")
    
    e = ddit.entropy("X")
    
    print(e)