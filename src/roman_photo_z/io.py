import os
import numpy as np
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

def organize(fname, output_dir) -> None:
    """Organizes files in output directory

    Parameters
    ----------
    fname: str
        name of resulting zip file

    output_dir: str
        Directory in which to zip images

    Returns
    -------
    None
    """
    if fname[-4:] != ".zip":
        fname += ".zip"
    save_path = os.path.join(output_dir, fname)
    if os.path.isfile(save_path):
        os.system(f"rm -rf {save_path}")
    if os.path.isdir(save_path[:-4]):
        os.system(f"rm -rf {save_path[:-4]}")
    cmd = f"zip -m -D -q {save_path} *.png *.log"
    os.system(cmd)

def load_filters() -> pd.DataFrame:
    """Loads filter transmission profiles

    Parameters
    ----------
    None

    Returns
    -------
    filters: pandas.DataFrame
        Filter transmission profiles as a function of wavelength
    """
    filters = pd.read_excel("https://roman.gsfc.nasa.gov/science/RRI/Roman_effarea_20210614.xlsx", skiprows=1)
    filters.columns = [column.strip() for column in filters.columns]
    if filters["Wave"][0] == 0:
        filters.drop(0, inplace=True)
    filters.drop(columns=["SNPrism", "Grism_1stOrder", "Grism_0thOrder"], inplace=True)
    # filters["Wave"] *= 10000 # micron to angstrom conversion
    filters[filters.columns.drop("Wave")] /= np.pi * 1.2**2 # effective area to throughput conversion
    return filters