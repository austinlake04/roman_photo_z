import os
import pandas as pd
import openpyxl

def save(database: dict[pd.DataFrame]) -> None:
    """Saves data to excel file

    Parameters
    ----------
    database: dict[pandas.DataFrame]
        Dictionary of dataframes for contains AB magnitudes for RST filters as a function of galaxy type and redshift 

    Returns
    -------
    None
    
    """
    global base_dir
    filename = os.path.join(base_dir, "output", "output.xlsx") # must exist
    wb = openpyxl.load_workbook(filename)
    writer = pd.ExcelWriter(filename, engine='openpyxl')
    for sheet, data in database.items():
        writer.sheets = dict((ws.title, ws) for ws in wb.worksheets) # need this to prevent overwrite
        data.to_excel(writer, index=False, sheet_name=sheet)
    writer.save()