import pandas as pd
from functools import partial
from bokeh.layouts import column, row
from bokeh.plotting import figure, ColumnDataSource
import numpy as np
from bokeh.models import Slider, Span, Button, Spinner
from bokeh.models.widgets import FileInput, TextInput
from io import BytesIO
from bokeh.server.server import Server
from base64 import b64decode
from utils import *
from data import *


def start(doc):
    def initialPlot(attr, old, new):
        decoded = b64decode(new)
        df = pd.read_csv(BytesIO(decoded), header=0, index_col=False)
        frames, values = obtainFrameValueLst(df)
        threshold = (min(values[100:]) + max(values[100:])) / 2
        derivative_values = derivative(values)
        time_max, max_val = getMax(frames, values, threshold)
        tStart, value_start, tEnd, value_end = startEndPeak(frames, values, derivative_values, threshold)

        TOOLTIPS = [("(x,y)", "($x, $y)")]
        tools1 = ['xwheel_zoom', 'xpan', 'reset']

        p1 = figure(title='Intensity over time', tools=tools1, active_scroll='xwheel_zoom', plot_width=1000,
                    plot_height=450, tooltips=TOOLTIPS)
        p2 = figure(title='Derivatives over time', tools=tools1, active_scroll='xwheel_zoom', plot_width=1000,
                    plot_height=450, tooltips=TOOLTIPS)
        p1.x_range = p2.x_range

        kernelSlider1 = Slider(title='Apply smoothing filter x times', start=0, end=4, step=1, value=0)
        kernelSlider2 = Slider(title='Smoothing filter width', start=3, end=13, step=2, value=3)
        cutSlider = Slider(title='Cut first x datapoints', start=0, end=100, step=1, value=0)
        cutSlider2 = Slider(title='Cut last x datapoints', start=0, end=100, step=1, value=0)
        cutSlider3 = Slider(title='Cut first x derivative', start=0, end=30, step=1, value=0)
        cutSlider4 = Slider(title='Cut last x derivative', start=0, end=30, step=1, value=0)
        fpsSpinner = Spinner(title="Enter framerate", step=50, value=600)
        text_input2 = TextInput(value="", title="Enter name output file without file extension")
        bt = Button(label='Click to save', height_policy='max')

        source = ColumnDataSource(data=dict(frames=frames, intensity=values))
        source2 = ColumnDataSource(data=dict(frames=frames, avgLine=values))
        source3 = ColumnDataSource(data=dict(frames=frames, dy=derivative_values))
        source4 = ColumnDataSource(data=dict(timeMaxima=time_max, maxima=max_val))
        source5 = ColumnDataSource(data=dict(timeStart=tStart, startValue=value_start))
        source6 = ColumnDataSource(data=dict(timeEnd=tEnd, endValue=value_end))
        settings = ColumnDataSource(data=dict(AvgFiltern=[kernelSlider1.value], AvgFilterWidth=[kernelSlider2.value],
                                              SkipInitial=[cutSlider.value], SkipLast=[cutSlider2.value]))

        hline = Span(location=threshold, dimension='width', line_color='green', line_width=3)
        p1.line('frames', 'intensity', line_alpha = .5, source=source)
        p2.line('frames', 'dy', line_color='blue', source=source3)
        p1.circle('timeMaxima', 'maxima', source=source4, fill_color='red', size=7)
        p1.circle('timeStart', 'startValue', source=source5, fill_color='green', size=7)
        p1.circle('timeEnd', 'endValue', source=source6, fill_color='purple', size=7)
        p1.renderers.extend([hline])

        rend = p1.line('frames', 'avgLine', source=source2, line_alpha=0, color='orange')

        def updateAvg(attr, old, new, frames, values):
            if cutSlider.value != 0 or cutSlider2.value != 0:
                if cutSlider2.value > 0:
                    maxInd = -cutSlider2.value - 1
                else:
                    maxInd = -1
                new_frames = frames[cutSlider.value:maxInd]
                new_values = values[cutSlider.value:maxInd]
            else:
                new_frames = frames
                new_values = values

            n = kernelSlider1.value
            val2smooth = np.array(new_values)
            if kernelSlider1.value != 0:
                while n != 0:
                    val2smooth = averageFilter(val2smooth, kernelSlider2.value)
                    n -= 1
                derivative_values = derivative(val2smooth)
                source2.data = {'frames': new_frames, 'avgLine': val2smooth}
                tStart, value_start, tEnd, value_end = startEndPeak(new_frames, new_values, derivative_values,
                                                                 threshold)
                source5.data = {'timeStart': tStart, 'startValue': value_start}
                source6.data = {'timeEnd': tEnd, 'endValue': value_end}


                if cutSlider4.value > 0:
                    maxInd = -cutSlider4.value - 1
                else:
                    maxInd = -1
                dy_frames = new_frames[cutSlider3.value: maxInd]
                new_dy = derivative_values[cutSlider3.value: maxInd]
                source3.data = {'frames': dy_frames, 'dy': new_dy}

                rend.glyph.line_alpha = 1
            else:
                rend.glyph.line_alpha = 0

            settings.data = {'AvgFiltern': [kernelSlider1.value], 'AvgFilterWidth': [kernelSlider2.value],
                             'SkipInitial': [cutSlider.value], 'SkipLast': [cutSlider2.value]}

        def cuttingPoints(attr, old, new, frames, values):
            if cutSlider2.value > 0: maxInd = -cutSlider2.value -1
            else: maxInd = -1
            new_frames = frames[cutSlider.value: maxInd]
            new_values = values[cutSlider.value: maxInd]
            yrange = max(new_values) - min(new_values)
            p1.y_range.start = min(new_values) - yrange / 10
            p1.y_range.end = max(new_values) + yrange / 10
            source.data = {'frames': new_frames, 'intensity': new_values}

            n = kernelSlider1.value
            val2smooth = np.array(new_values)
            if kernelSlider1.value != 0:
                while n != 0:
                    val2smooth = averageFilter(val2smooth, kernelSlider2.value)
                    n -= 1
                    derivative_values = derivative(val2smooth)
            else:
                derivative_values = derivative(new_values)
            new_tStart, nvalue_start, new_tEnd, nvalue_end = startEndPeak(new_frames, new_values, derivative_values,
                                                                       threshold)

            source2.data = {'frames': new_frames, 'avgLine': val2smooth}
            source5.data = {'timeStart': new_tStart, 'startValue': nvalue_start}
            source6.data = {'timeEnd': new_tEnd, 'endValue': nvalue_end}

            if cutSlider4.value > 0: maxInd = -cutSlider4.value -1
            else: maxInd = -1
            dy_frames = new_frames[cutSlider3.value: maxInd]
            new_dy = derivative_values[cutSlider3.value: maxInd]

            source3.data = {'frames': dy_frames, 'dy': new_dy}
            settings.data = {'AvgFiltern': [kernelSlider1.value], 'AvgFilterWidth': [kernelSlider2.value],
                             'SkipInitial': [cutSlider.value], 'SkipLast': [cutSlider2.value]}

        def cuttingDerivative(attr, old, new, frames, values):
            if cutSlider2.value > 0: maxInd = -cutSlider2.value -1
            else: maxInd = -1
            new_frames = frames[cutSlider.value: maxInd]
            new_values = values[cutSlider.value: maxInd]

            n = kernelSlider1.value
            val2smooth = np.array(new_values)
            if kernelSlider1.value != 0:
                while n != 0:
                    val2smooth = averageFilter(val2smooth, kernelSlider2.value)
                    n -= 1
                    derivative_values = derivative(val2smooth)
            else:
                derivative_values = derivative(new_values)

            if cutSlider4.value > 0: maxInd = -cutSlider4.value -1
            else: maxInd = -1
            dy_frames = new_frames[cutSlider3.value: maxInd]
            new_dy = derivative_values[cutSlider3.value: maxInd]

            source3.data = {'frames': dy_frames, 'dy': new_dy}

        kernelSlider1.on_change('value', partial(updateAvg, frames=frames, values=values))
        kernelSlider2.on_change('value', partial(updateAvg, frames=frames, values=values))
        cutSlider.on_change('value', partial(cuttingPoints, frames=frames, values=values))
        cutSlider2.on_change('value', partial(cuttingPoints, frames=frames, values=values))
        cutSlider3.on_change('value', partial(cuttingDerivative, frames=frames, values=values))
        cutSlider4.on_change('value', partial(cuttingDerivative, frames=frames, values=values))
        bt.on_click(partial(save, source, source4, source5, source6, fpsSpinner.value, settings))

        layout2 = row(column(p1, p2), column(kernelSlider1, kernelSlider2, cutSlider, cutSlider2, cutSlider3,
                                             cutSlider4, fpsSpinner, row(text_input2, bt)))

        doc.remove_root(layout)
        doc.add_root(layout2)

    fileInp = FileInput(accept=".csv")
    fileInp.on_change('value', initialPlot)
    layout = column(fileInp)
    doc.add_root(layout)


server = Server({'/': start}, num_procs=1)
server.start()

if __name__ == '__main__':
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
