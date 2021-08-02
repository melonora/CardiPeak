import pandas as pd
from bokeh.io import export_png
from typing import Tuple, List
from utils import frameTime, getAmplitudes
import os
import openpyxl


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


def getOutputDirs(root_dir: str) -> List:
    """
    Function to check if output directory exists and to check the directories within an output directory. If output
    directory does not exists, one is created.

    Parameters
    ----------
    root_dir : str
        String indicating the path of the output directory

    Returns
    -------
    List
        List containing the directory names in root_dir
    """
    if os.path.exists(root_dir):
        return [i for i in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, i))]
    else:
        os.mkdir(root_dir)
        return []


def save(analyzed_data, max_data, start_data, end_data, settings, output_data, plot1, plot2, raw_data):
    fps = settings.data['fps'][0]
    subDir = output_data.data['output_dir'][0]
    outputFile = output_data.data['output_file'][0]
    extension = output_data.data['ext'][0]
    lsData = list(zip(analyzed_data.data['frames'], analyzed_data.data['intensity']))
    lsMax = frameTime(list(zip(max_data.data['timeMaxima'], max_data.data['maxima'],
                               ['max'] * len(max_data.data['maxima']), max_data.data['set'])), fps)
    lsStart = frameTime(list(zip(start_data.data['timeStart'], start_data.data['startValue'],
                                 ['start'] * len(start_data.data['startValue']), start_data.data['set'])), fps)
    lsEnd = frameTime(list(zip(end_data.data['timeEnd'], end_data.data['endValue'],
                               ['end'] * len(end_data.data['endValue']), end_data.data['set'])), fps)
    lsPoints = sorted(lsMax + lsStart + lsEnd)

    startIndex = -1
    point = "end"
    for i in range(len(lsPoints)):
        endIndex = lsData.index(lsPoints[i][:2])
        lsData[endIndex] += (lsPoints[i][2], lsPoints[i][3],)
        if lsPoints[i][2] == 'end':
            for i in range(startIndex+1, endIndex):
                lsData[i] += ('maxEnd', None,)
                point = 'end'
        elif lsPoints[i][2] == 'max':
            for i in range(startIndex+1, endIndex):
                lsData[i] += ('startMax', None,)
                point = 'max'
        else:
            for i in range(startIndex+1, endIndex):
                lsData[i] += ('endStart', None,)
                point = 'start'
        startIndex = endIndex

    if point == 'end':
        for i in range(startIndex+1, len(lsData)):
            lsData[i] += ('endStart', None,)
    elif point == 'max':
        for i in range(startIndex+1, len(lsData)):
            lsData[i] += ('maxEnd', None,)
    else:
        for i in range(startIndex+1, len(lsData)):
            lsData[i] += ('startMax', None,)

    lsData = frameTime(lsData, fps)
    peakMaxInterval = [lsMax[i + 1][-1] - lsMax[i][-1] for i in range(len(lsMax) - 1)]
    bgInterval = [lsPoints[i+1][-1] - lsPoints[i][-1] for i in range(len(lsPoints)-1) if lsPoints[i][2] == 'end']
    amplitudes = getAmplitudes(lsPoints)
    startMaxTime = [lsPoints[i+1][-1] - lsPoints[i][-1] for i in range(len(lsPoints)-1) if lsPoints[i][2] == 'start']
    maxEndTime = [lsPoints[i+1][-1] - lsPoints[i][-1] for i in range(len(lsPoints) - 1) if lsPoints[i][2] == 'max']
    peakTime = [lsPoints[i+2][-1] - lsPoints[i][-1] for i in range(len(lsPoints)-2) if lsPoints[i][2] == 'start']

    pdData = pd.DataFrame({'Frame_index': [i[0] for i in lsData], 'Time_(s)': [i[4] for i in lsData],
                           'Intensity': [i[1] for i in lsData], 'Type': [i[2] for i in lsData],
                           'set': [i[3] for i in lsData]})
    pdBgInterval = pd.DataFrame({'background_interval': bgInterval})
    pdPeakMaxInterval = pd.DataFrame({'Peak_max_interval': peakMaxInterval})
    pdStartMaxT = pd.DataFrame({'start_max(s)': startMaxTime})
    pdMaxEndT = pd.DataFrame({'max_end(s)': maxEndTime})
    pdPeakTime = pd.DataFrame({'Peak_time(s)': peakTime})
    pdAmplitudes = pd.DataFrame({'Peak_amplitude': amplitudes})
    pdSettings = pd.DataFrame(settings.data)
    pd_complete = pd.concat([pdData, pdBgInterval, pdPeakMaxInterval, pdStartMaxT, pdMaxEndT, pdPeakTime, pdAmplitudes,
                             pdSettings], ignore_index=True, axis=1)
    pd_complete.columns = ['Frame_index', 'Time_(s)', 'Intensity', 'Type', 'set', 'background_interval',
                           'Peak_max_interval', 'Start_max(s)', 'Max_end(s)', 'Peak_time(s)', 'Peak_amplitude'] +\
                          [i for i in settings.data]

    outputDir = '../output'
    if not os.path.exists(os.path.join(outputDir, subDir)):
        os.makedirs(os.path.join(outputDir, subDir))
    if subDir != "please overwrite":
        if outputFile == '':
            pd_complete.to_csv(os.path.join(outputDir, subDir, 'output.csv'), index=False)
        elif extension == '.csv':
            pd_complete.to_csv(os.path.join(outputDir, subDir, outputFile + extension), index=False)
        elif extension == '.xlsx':
            export_png(plot1, filename=os.path.join(outputDir, subDir, 'image.png'))
            export_png(plot2, filename=os.path.join(outputDir, subDir, 'image2.png'))
            with pd.ExcelWriter(os.path.join(outputDir, subDir, outputFile + extension)) as writer:
                pd_complete.to_excel(writer, index=False, sheet_name='Analysis_results')
                raw_data.to_excel(writer, index=False, sheet_name='Raw_data')

            workbook = openpyxl.load_workbook(os.path.join(outputDir, subDir, outputFile + extension))
            ws1 = workbook.create_sheet("Plots")
            img = openpyxl.drawing.image.Image(os.path.join(outputDir, subDir, 'image.png'))
            img.anchor = 'B2'
            img2 = openpyxl.drawing.image.Image(os.path.join(outputDir, subDir, 'image2.png'))
            img2.anchor = 'B27'
            ws1.add_image(img)
            ws1.add_image(img2)
            workbook.save(os.path.join(outputDir, subDir, outputFile + extension))
            os.remove(os.path.join(outputDir, subDir, 'image.png'))
            os.remove(os.path.join(outputDir, subDir, 'image2.png'))
