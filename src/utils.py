import numpy as np
from typing import List, Tuple, Optional


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
    halfFloored = int(avgFilter.shape[0] / 2)
    for i in range(halfFloored, averageY.shape[0]-halfFloored):
        averageY[i] = (y[i-halfFloored:i+halfFloored+1] * avgFilter).sum()
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


def getMax(frames: List[int], values: List[float], avgValues: List[float], threshold: int)\
        -> Tuple[List[int], List[float]]:
    """ Function to retrieve the maximum value of a sequence of values above a given threshold and the specific frame at
    which this maximum occurs. Maximum values are obtained by iterating over a list of values, determining the start and
    end index of the sequence about a given threshold and then determining the index of the maximum value in
    AvgValues[start: end]. The start index is added to obtain the index in the full list. The frame and real value from
    values is obtained with this index.

    Parameters
    ----------
    frames : List[int]
        List of integers in which each integer corresponds to the index of a frame.
    values : List[float]
        List of floats in which each float corresponds to a calcium intensity of a given frame.
    avgValues : List[float]
        List of floats resulting from an average filter applied to the values list.
    threshold : float
        Threshold equal to a float value used to determine which parts of a list of float values are above a given
        threshold.

    Returns
    -------
    frameMaxValues: List[int]
        List containing the indexes of frames with the maximum calcium intensity value of peaks.
    maxValues: List[float]
        List of float values corresponding to each maximum intensity value of each calcium intensity peak in the values
        list.
    """
    if values[0] > threshold:
        above = True
    else:
        above = False
    frameMaxValues = []
    maxValues = []
    indexAbove = []
    frameMinValues = []
    minValues = []
    indexBelow = []

    for i in range(len(values)):
        if above:
            if values[i] <= threshold:
                minIndex, maxIndex = indexAbove[0], indexAbove[-1]
                if len(values[minIndex:maxIndex]) >= 5:
                    index = avgValues[minIndex:maxIndex].index(max(avgValues[minIndex:maxIndex])) + minIndex
                    frameMaxValues.append(frames[index])
                    maxValues.append(values[index])
                above = False
                indexBelow = [i]
            else:
                indexAbove.append(i)
        else:
            if values[i] < threshold:
                indexBelow.append(i)
            else:
                minIndex, maxIndex = indexBelow[0], indexBelow[-1]
                if len(values[minIndex:maxIndex]) >= 5:
                    index = avgValues[minIndex:maxIndex].index(min(avgValues[minIndex:maxIndex])) + minIndex
                    frameMinValues.append(frames[index])
                    minValues.append(values[index])
                indexAbove = [i]
                above = True
    return frameMaxValues, maxValues, frameMinValues, minValues


def addTimeValue(framePointLs: List[Optional[int]], valuePointLs: List[Optional[float]], frameLs: List[int],
                 valueLs: List[float], index: int):
    """ Function appending the frame and the corresponding intensity value of that frame to the corresponding lists
    given an index.

    Parameters
    ----------
    framePointLs: List[Optional[int]]
        List containing integers corresponding to the frames of either starts, max or end of the peaks.
    valuePointLs: List[Optional[float]]
        List containing float elements corresponding to the values at starts, max or end of the peaks.
    frameLs: List[int]
        List containing integers corresponding to the frame numbers.
    valueLs: List[float]
        List with float elements corresponding to the calcium intensities.
    index: int
        Integer indicating the index of either the start, max or end of a peak.

    Returns
    -------
    framePointLs: List[Optional[int]]
        List containing integers corresponding to the frames of either starts, max or end of the peaks.
    valuePointLs: List[Optional[float]]
        List containing float elements corresponding to the values at starts, max or end of the peaks.
    """
    if valueLs[index] < valueLs[index + 1]:
        framePointLs.append(frameLs[index])
        valuePointLs.append(valueLs[index])
    else:
        framePointLs.append(frameLs[index + 1])
        valuePointLs.append(valueLs[index + 1])
    return framePointLs, valuePointLs


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
                            timeStartPeaks, valueStartPeaks = addTimeValue(timeStartPeaks, valueStartPeaks, frames,
                                                                           values, t)
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
                            timeEndPeaks, valueEndPeaks = addTimeValue(timeEndPeaks, valueEndPeaks, frames,
                                                                           values, t)
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
    frames: List[Tuple[int, float, str, float]]
        List of tuples where the last float is the time.
    """
    for i in range(len(frames)):
        frames[i] += (1. / fps * frames[i][0],)
    return frames


def getAmplitudes(peakPoints: List[Tuple[int, float, str, float]]) -> List[float]:
    """ Function returning a list of peak amplitudes. The peak amplitudes are calculated as the average of the
    difference between the max value of the peak and the value at the start of the peak and the max value of the peak
    and the value at the end of the peak.

    Parameters
    ----------
    peakPoints: List[Tuple[int, float, str, float]]
        List of tuples where each tuple contains information regarding either a start point, max point or end point of
        a peak. The first element corresponds to the frame index, the second element to the intensity value, the third
        element indicates whether the point is either the start, max or end of a peak and the fourth element indicates
        the time after 0 at which the frame was taken.

    Returns
    -------
    amplitudes: List[float]
        A list of floats where each float corresponds to the peak amplitude of a peak. The elements are in chronological
        order.
    """
    amplitudes = []
    for i in range(len(peakPoints)):
        if peakPoints[i][2] == 'start':
            try:
                amplitudes.append((peakPoints[i+1][1] - peakPoints[i][1] + peakPoints[i+1][1] - peakPoints[i+2][1]) / 2)
            except IndexError:
                pass
    return amplitudes


def getAmplitudes2(peakPoints: List[Tuple[int, float, str, float]]) -> List[float]:
    """ Function returning a list of peak amplitudes. The peak amplitudes are calculated as the average of the
    difference between the max value of the peak and the value at the start of the peak and the max value of the peak
    and the value at the end of the peak.

    Parameters
    ----------
    peakPoints: List[Tuple[int, float, str, float]]
        List of tuples where each tuple contains information regarding either a start point, max point or end point of
        a peak. The first element corresponds to the frame index, the second element to the intensity value, the third
        element indicates whether the point is either the start, max or end of a peak and the fourth element indicates
        the time after 0 at which the frame was taken.

    Returns
    -------
    amplitudes: List[float]
        A list of floats where each float corresponds to the peak amplitude of a peak. The elements are in chronological
        order.
    """
    amplitudes = []
    for i in range(len(peakPoints)):
        if peakPoints[i][2] == 'min':
            try:
                amplitudes.append((peakPoints[i+1][1] - peakPoints[i][1] + peakPoints[i+1][1] - peakPoints[i+2][1]) / 2)
            except IndexError:
                pass
    return amplitudes


def calc_threshold(values, percentage=50):
    """ Function to select a threshold based on a list of values where the lowest threshold is equal to the minimum
    value and the largest threshold is equal to the maximum value.

    Parameters
    ----------
    values
    percentage

    Returns
    -------

    """
    return min(values[100:]) + (max(values[100:]) - min(values[100:])) * percentage / 100
