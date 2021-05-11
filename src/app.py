import pandas as pd
from functools import partial
from bokeh.layouts import column, row
from bokeh.plotting import figure, ColumnDataSource
import numpy as np
from bokeh.models import Slider, Span, CrosshairTool
from bokeh.models.widgets import FileInput
from io import BytesIO
from bokeh.server.server import Server
from base64 import b64decode
from utils import *


def start(doc):
    def initialPlot(attr, old, new):
        decoded = b64decode(new)
        df = pd.read_csv(BytesIO(decoded), header=0, index_col=False)
        frames, values = obtainFrameValueLst(df)
        threshold = (min(values[100:]) + max(values[100:])) / 2
        derivative_values = derivative(values)
        derivative2_values = derivative(derivative_values)
        time_max, max_val = getMax(frames, values, threshold)
        tStart, value_start, tEnd, value_end = startPeak(frames, values, derivative_values, threshold)

        source = ColumnDataSource(data=dict(frames=frames, intensity=values))
        source2 = ColumnDataSource(data=dict(frames=frames, avgLine=values))
        source3 = ColumnDataSource(data=dict(frames=frames, dy=derivative_values))
        source4 = ColumnDataSource(data=dict(frames=frames, dy2=derivative2_values))
        source5 = ColumnDataSource(data=dict(timeMaxima=time_max, maxima=max_val))
        source6 = ColumnDataSource(data=dict(timeStart=tStart, startValue=value_start))
        source7 = ColumnDataSource(data=dict(timeEnd=tEnd, endValue=value_end))

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
        spanSlider = Slider(title='Span for minimum around time max dy peak', start=0, end=30, step=1, value=0)

        hline = Span(location=threshold, dimension='width', line_color='green', line_width=3)
        p1.line('frames', 'intensity', source=source)
        p2.line('frames', 'dy', line_color='blue', source=source3)
        p2.line('frames', 'dy2', line_color='red', source=source4)
        p1.circle('timeMaxima', 'maxima', source=source5, fill_color='red', size=7)
        p1.circle('timeStart', 'startValue', source=source6, fill_color='green', size=7)
        p1.circle('timeEnd', 'endValue', source=source7, fill_color='purple', size=7)
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
                derivative2_values = derivative(derivative_values)
                source2.data = {'frames': new_frames, 'avgLine': val2smooth}

                if cutSlider4.value > 0:
                    maxInd = -cutSlider4.value - 1
                else:
                    maxInd = -1
                new_frames = new_frames[cutSlider3.value: maxInd]
                new_dy = derivative_values[cutSlider3.value: maxInd]
                new_dy2 = derivative2_values[cutSlider3.value: maxInd]
                new_tStart, nvalue_start, new_tEnd, nvalue_end = startPeak(new_frames, new_values, derivative_values,
                                                                           threshold)
                source3.data = {'frames': new_frames, 'dy': new_dy}
                source4.data = {'frames': new_frames, 'dy2': new_dy2}
                source6.data = {'timeStart': new_tStart, 'startValue': nvalue_start}
                source7.data = {'timeEnd': new_tEnd, 'endValue': nvalue_end}
                rend.glyph.line_alpha = 1
            else:
                rend.glyph.line_alpha = 0

        def cuttingPoints(attr, old, new, frames, values):
            if cutSlider2.value > 0: maxInd = -cutSlider2.value -1
            else: maxInd = -1
            new_frames = frames[cutSlider.value: maxInd]
            new_values = values[cutSlider.value: maxInd]
            p1.y_range.start = min(new_values) - threshold / 20
            p1.y_range.end = max(new_values) + threshold / 20
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

            derivative2_values = derivative(derivative_values)


            source2.data = {'frames': new_frames, 'avgLine': val2smooth}



            if cutSlider4.value > 0: maxInd = -cutSlider4.value -1
            else: maxInd = -1
            new_frames = new_frames[cutSlider3.value: maxInd]
            new_dy = derivative_values[cutSlider3.value: maxInd]
            new_dy2 = derivative2_values[cutSlider3.value: maxInd]
            new_tStart, nvalue_start, new_tEnd, nvalue_end = startPeak(new_frames, new_values, new_dy,
                                                                       threshold)
            source3.data = {'frames': new_frames, 'dy': new_dy}
            source4.data = {'frames': new_frames, 'dy2': new_dy2}
            source6.data = {'timeStart': new_tStart, 'startValue': nvalue_start}
            source7.data = {'timeEnd': new_tEnd, 'endValue': nvalue_end}

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
                    dy = derivative(val2smooth)
            else:
                dy = derivative(new_values)
            dy2 = derivative(dy)

            if cutSlider4.value > 0: maxInd = -cutSlider4.value -1
            else: maxInd = -1
            new_frames = new_frames[cutSlider3.value: maxInd]
            dy = dy[cutSlider3.value: maxInd]
            dy2 = dy2[cutSlider3.value: maxInd]
            new_tStart, nvalue_start, new_tEnd, nvalue_end = startPeak(new_frames, new_values, dy,
                                                                       threshold)

            source3.data = {'frames': new_frames, 'dy': dy}
            source4.data = {'frames': new_frames, 'dy2': dy2}
            source7.data = {'timeEnd': new_tEnd, 'endValue': nvalue_end}

        kernelSlider1.on_change('value', partial(updateAvg, frames=frames, values=values))
        kernelSlider2.on_change('value', partial(updateAvg, frames=frames, values=values))
        cutSlider.on_change('value', partial(cuttingPoints, frames=frames, values=values))
        cutSlider2.on_change('value', partial(cuttingPoints, frames=frames, values=values))
        cutSlider3.on_change('value', partial(cuttingDerivative, frames=frames, values=values))
        cutSlider4.on_change('value', partial(cuttingDerivative, frames=frames, values=values))

        layout2 = row(column(p1, p2), column(kernelSlider1, kernelSlider2, cutSlider, cutSlider2, cutSlider3,
                                             cutSlider4, spanSlider))

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
