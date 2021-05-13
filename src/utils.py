import numpy as np
from typing import List, Tuple


def averageFilter(y: List[float], filterWidth: int) -> np.array:
    """ Function applying an average smoothing filter with a given filter width to a list in which each element is a
    float (decimal number).

    Parameters
    ----------
    y : List[float]
        Python list in which each element is a float and to which the average smoothing filter will be applied.
    filterWidth : int
        The width of the smoothing filter to be applied. Should always be an unequal number to allow for the output
        array to be of equal length as the input list of float values.

    Returns
    -------
    np.array:
        A np.array containing a sequence of float values which is the result of the applied average smoothing filter
        applied to the input list of float elements.
    """
    avgFilter = np.ones(filterWidth) / filterWidth
    averageY = np.array(y)
    halfMinusOne = int(avgFilter.shape[0] / 2)
    for i in range(halfMinusOne, averageY.shape[0]-halfMinusOne):
        averageY[i] = (y[i-halfMinusOne:i+halfMinusOne+1] * avgFilter).sum()
    return averageY


def derivative(values: List[float]) -> List[float]:
    """ Function taking a list of floats with length i and returns a list of length i where each element is the
    difference between input[i+1] and input[i]. The last value of the list to be returned is appended another time.
    to ensure equal lengths of the input list and the list to be returned.

    Parameters
    ----------
    values : List[float]
        List in which each element is a float.

    Returns
    -------
    dy: List[float]
        List of the same length as the values list in which each element is a float corresponding to the change in two
        subsequent values in the values list.
    """
    dy = [values[i+1] - values[i] for i in range(len(values)-1)]
    dy.append(dy[-1])
    return dy


def getMax(frames: List[int], values: List[float], threshold: int) -> Tuple[List[int], List[float]]:
    """ Function to retrieve the maximum value of a sequence of values above a given threshold and the specific frame at
    which this maximum occurs.

    Parameters
    ----------
    frames : List[int]
        List of integers in which each integer corresponds to the index of a frame.
    values : List[float]
        List of floats in which each float corresponds to a calcium intensity of a given frame.
    threshold : float
        Threshold equal to a float value used to determine which parts of a list of float values are above a given
        threshold.

    Returns
    -------
    timeMax: List[int]
        List containing the indexes of frames with the maximum calcium intensity value of peaks.
    maxVals: List[float]
        List of float values corresponding to each maximum intensity value of each calcium intensity peak in the values
        list.
    """
    if values[0] > threshold:
        above = True
    else:
        above = False
    timeMax = []
    maxVals = []
    indexAbove = []

    for i in range(len(values)):
        if above:
            if values[i] <= threshold:
                minIndex, maxIndex = indexAbove[0], indexAbove[-1]
                if len(values[minIndex:maxIndex]) >= 20:
                    timeMax.append(frames[values[minIndex:maxIndex].index(max(values[minIndex:maxIndex])) + minIndex])
                    maxVals.append(max(values[minIndex:maxIndex]))

                above = False
            else:
                indexAbove.append(i)
        else:
            if values[i] < threshold:
                pass
            else:
                indexAbove = [i]
                above = True
    return timeMax, maxVals


def startEndPeak(frames: List[int], values: List[float], derivative: List[float],
                 threshold: int) -> Tuple[List[int], List[float], List[int], List[float]]:
    """ Function to obtain the indexes of frames and the corresponding intensity values of the start and end of calcium
    intensity peaks.

    Parameters
    ----------
    frames : List[int]
        List of integers in which each integer corresponds to the index of a frame.
    values : List[float]
        List of floats in which each float corresponds to a calcium intensity of a given frame.
    derivative : List[Float]
        List of float values corresponding to the difference in subsequent values in the values list.
    threshold : float
        Threshold equal to a float value used to determine which parts of a list of float values are below a given
        threshold.

    Returns
    -------
    timeStartPeaks: List[int]
        List of integers corresponding to the frames where a calcium intensity peak starts.
    valueStartPeaks: List[float]
        List of float values corresponding to the calcium intensity values at the start of calcium intensity peaks.
    timeEndPeaks: List[int]
        List of integers corresponding to the frames where a calcium intensity peak ends.
    valueEndPeaks: List[float]
        List of float values corresponding to the calcium intensity values at the end of calcium intensity peaks.
    """
    if values[0] < threshold:
        below = True
    else:
        below = False
    timeStartPeaks = []
    valueStartPeaks = []
    timeEndPeaks = []
    valueEndPeaks = []
    indexBelow = []

    for i in range(len(values)):
        if below:
            if values[i] <= threshold:
                indexBelow.append(i)
            else:
                subSlice = slice(indexBelow[0], indexBelow[-1])
                diff = indexBelow[-1] - indexBelow[0]
                if len(values[subSlice]) >= 20:
                    max_dyFrameIndex = derivative[subSlice].index(max(derivative[subSlice])) + indexBelow[0]
                    for t in range(max_dyFrameIndex, max(max_dyFrameIndex-diff, 0), -1):
                        if derivative[t] > 0:
                            pass
                        else:
                            if values[t] < values[t+1]:
                                timeStartPeaks.append(frames[t])
                                valueStartPeaks.append(values[t])
                            else:
                                timeStartPeaks.append(frames[t+1])
                                valueStartPeaks.append(values[t+1])
                            break
                below = False
        else:
            if values[i] >= threshold:
                pass
            else:
                indexBelow = [i]
                if all([i < threshold for i in values[i:min(i+100, len(values))]]) and \
                        len(derivative[i:min(i+100, len(derivative))]) > 20:
                    min_dyFrameIndex = derivative[i:i+100].index(min(derivative[i:i+100])) + i
                    for t in range(min_dyFrameIndex, len(derivative)-1):
                        if derivative[t] < 0:
                            pass
                        else:
                            if values[t] < values[t+1]:
                                timeEndPeaks.append(frames[t])
                                valueEndPeaks.append(values[t])
                            else:
                                timeEndPeaks.append(frames[t+1])
                                valueEndPeaks.append(values[t+1])
                            break
                below = True
    return timeStartPeaks, valueStartPeaks, timeEndPeaks, valueEndPeaks


def frameTime(frames: List[Tuple[int, float, str]], fps: int) -> List[Tuple[int, float, str, float]]:
    """ Function taking a list of integers corresponding to the indexes of frames and converting them to times using
    fps.

    Parameters
    ----------
    frames: List[Tuple[int, float, str]]
        List of tuples with an integer representing the frame index to be converted to a time.
    fps: int
        Framerate per second indicated by an integer.

    Returns
    -------
    List[Tuple[int, float, str, float]]
        List of tuples where the last float is the time.
    """
    for i in range(len(frames)):
        frames[i] += (1. / fps * frames[i][0],)
    return frames
