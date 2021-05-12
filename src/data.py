import pandas as pd
from typing import Tuple, List


def obtainFrameValueLst(df: pd.DataFrame) -> Tuple[List[int], List[float]]:
    """ Function to retrieve the columns of a pandas dataframe corresponding to the frames and the pixel value of each
    frame.

    Parameters
    ----------
    df : pd.DataFrame
        Pandas dataframe containing the frames column and the pixel value column as a minimum.

    Returns
    -------
    Tuple[List[int], List[float]]
        Tuple containing two lists where the first lists corresponds to the column containing the subsequent indexes of
        frames and the second list contains the pixel intensity value of each frame.
    """
    if len({'Frame', 'Pixel value'}.intersection(set(df.columns))) == 2:
        frames = df['Frame']
        values = df['Pixel value']
    else:
        frames = df.iloc[:, 0]
        values = df.iloc[:, 1]
    return frames.tolist(), values.tolist()


def save(raw_data, max_data, start_data, end_data, fps):
    pass
