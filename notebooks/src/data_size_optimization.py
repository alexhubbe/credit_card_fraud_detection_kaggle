import pandas as pd

def downcast_columns(df):
    """
    Downcast numeric columns in the DataFrame: 
    - Columns with all integer values are downcasted to the most efficient integer type.
    - Columns with decimal values are downcasted to the most efficient float type.

    Args:
    - df (pandas.DataFrame): The DataFrame whose numeric columns will be downcasted.

    Returns:
    - pandas.DataFrame: The DataFrame with downcasted columns.
    """
    # Loop through numeric columns and check if all values are integers
    for column in df.select_dtypes(include='number'):
        # Check if all values in the column are integers using .is_integer()
        all_integers = df[column].apply(float).apply(lambda x: x.is_integer()).all()
        
        print(f"Column: {column}, All integers: {all_integers}")

        # Downcast based on whether all values are integers or not
        if all_integers:
            df[column] = pd.to_numeric(df[column], downcast='integer')  # Downcast to integer
        else:
            df[column] = pd.to_numeric(df[column], downcast='float')  # Downcast to float