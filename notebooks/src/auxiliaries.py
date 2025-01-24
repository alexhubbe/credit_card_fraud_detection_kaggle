import pandas as pd


def dataframe_coefficients(coefs, columns):
    return pd.DataFrame(data=coefs, index=columns, columns=["coefficient"]).sort_values(
        by="coefficient"
    )
