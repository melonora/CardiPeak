import pandas as pd

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
    frames = df['Frame']
    values = df['Pixel value']
    return frames.tolist(), values.tolist()
