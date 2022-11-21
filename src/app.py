from functools import partial
from bokeh.layouts import column, row
from bokeh.plotting import figure, ColumnDataSource
from bokeh.models import Slider, Span, Button, Spinner, TapTool, Select, PointDrawTool
from bokeh.models.widgets import FileInput, TextInput, RadioButtonGroup, Tabs, Panel
from bokeh.server.server import Server
from utils import *
from data import *


def start(doc):
    """
    Function to st
    """
    def initialPlot(attr, old, new):
        df = base64_to_df(new)
        frames, values = obtainFrameValueLst(df)
        init_smooth = averageFilter(values, 3)
        threshold = calc_threshold(init_smooth)
        derivative_values = derivative(values)
        time_max, max_val, time_min, min_val = getMax(frames, values, values, threshold)
        tStart, value_start, tEnd, value_end = startEndPeak(frames, values, derivative_values, threshold)
        outputDirs = getOutputDirs('../output')

        TOOLTIPS = [("(x,y)", "($x, $y)")]
        tools1 = ['xwheel_zoom', 'xpan', 'reset']

        p1 = figure(title='Intensity over time', tools=tools1, active_scroll='xwheel_zoom', plot_width=1000,
                    plot_height=450, tooltips=TOOLTIPS)
        p2 = figure(title='Derivative over time', tools=tools1, active_scroll='xwheel_zoom', plot_width=1000,
                    plot_height=450, tooltips=TOOLTIPS)
        p3 = figure(title='Intensity over time', tools=tools1, active_scroll='xwheel_zoom', plot_width=1000,
                    plot_height=450, tooltips=TOOLTIPS)
        p1.x_range = p2.x_range

        kernelSlider1 = Slider(title='Apply smoothing filter x times', start=0, end=8, step=1, value=0)
        kernelSlider2 = Slider(title='Smoothing filter width', start=3, end=13, step=2, value=3)
        cutSlider = Slider(title='Cut first x datapoints', start=0, end=100, step=1, value=0)
        cutSlider2 = Slider(title='Cut last x datapoints', start=0, end=100, step=1, value=0)
        cutSlider3 = Slider(title='Cut first x derivative', start=0, end=30, step=1, value=0)
        cutSlider4 = Slider(title='Cut last x derivative', start=0, end=30, step=1, value=0)
        valueSlider = Slider(title='Selected datapoint index', start=0, end=len(frames), value=0, margin=(0, 30, 0, 40))
        thresholdSpinner = Spinner(title="Threshold level relative to minimum value", low=0, high=100, step=5, value=50)
        fpsSpinner = Spinner(title="Enter framerate", step=50, value=600)
        text_input = TextInput(value_input="", title="Enter name output file without file extension")
        text_input2 = TextInput(value_input="", title="Enter name of new output directory",
                                placeholder="please overwrite")
        ext = RadioButtonGroup(labels=['.csv', '.xlsx'], orientation='vertical', height_policy='min', active=0)
        bt = Button(label='Click to save', height_policy='max')
        bt2 = Button(label='Click to save', height_policy='max')
        bt3 = Button(label='Click to create directory', height_policy='max')
        fileInp2 = FileInput(accept=".csv")
        selectDir = Select(title='output directory', value='new', options=['new']+outputDirs, width_policy='min')

        source1 = ColumnDataSource(data=dict(frames=frames, intensity=values))
        source2 = ColumnDataSource(data=dict(frames=frames, avgLine=values))
        source3 = ColumnDataSource(data=dict(frames=frames, dy=derivative_values))
        source4 = ColumnDataSource(data=dict(timeMaxima=time_max, maxima=max_val,
                                             set=["auto" for i in range(len(max_val))]))
        source5 = ColumnDataSource(data=dict(timeStart=tStart, startValue=value_start,
                                             set=["auto" for i in range(len(value_start))]))
        source6 = ColumnDataSource(data=dict(timeEnd=tEnd, endValue=value_end,
                                             set=["auto" for i in range(len(value_end))]))
        source7 = ColumnDataSource(data=dict(timeMinima=time_min, minima=min_val,
                                             set=["auto" for i in range(len(min_val))]))
        output = ColumnDataSource(data=dict(output_dir=[selectDir.value], output_file=[text_input2.value],
                                            ext=[ext.labels[ext.active]]))
        settings = ColumnDataSource(data=dict(AvgFiltern=[kernelSlider1.value], AvgFilterWidth=[kernelSlider2.value],
                                              SkipInitial=[cutSlider.value], SkipLast=[cutSlider2.value],
                                              ThresholdLevel=[thresholdSpinner.value], fps=[fpsSpinner.value]))

        hline = Span(location=threshold, dimension='width', line_color='green', line_width=3)
        p1.line('frames', 'intensity', line_alpha=.5, source=source1)
        p2.line('frames', 'dy', line_color='blue', source=source3)
        p3.line('frames', 'intensity', line_alpha=.5, source=source1)
        max_rend = p1.circle('timeMaxima', 'maxima', source=source4, fill_color='red', size=7)
        max_rend2 = p3.circle('timeMaxima', 'maxima', source=source4, fill_color='red', size=7)
        min_rend = p3.circle('timeMinima', 'minima', source=source7, fill_color='green', size=7)
        start_rend = p1.circle('timeStart', 'startValue', source=source5, fill_color='green', size=7)
        end_rend = p1.circle('timeEnd', 'endValue', source=source6, fill_color='purple', size=7)
        p1.renderers.extend([hline])
        p3.renderers.extend([hline])
        p1rend = [max_rend, start_rend, end_rend]
        p3rend = [max_rend2, min_rend]
        p1.add_tools(TapTool(renderers=p1rend))
        p1.add_tools(PointDrawTool(renderers=[start_rend], description="Add start peak"))
        p1.add_tools(PointDrawTool(renderers=[max_rend], description="Add max peak"))
        p1.add_tools(PointDrawTool(renderers=[end_rend], description="Add end peak"))
        p3.add_tools(TapTool(renderers=p3rend), PointDrawTool(renderers=p3rend))

        rend = p1.line('frames', 'avgLine', source=source2, line_alpha=0, color='orange')
        rend2 = p3.line('frames', 'avgLine', source=source2, line_alpha=0, color='orange')

        def updateAvg(attr, old, new, frames, values):
            """
            Parameters
            ----------

            Returns
            -------

            """
            threshold = calc_threshold(init_smooth, thresholdSpinner.value)
            hline.location = threshold
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
                if cutSlider4.value > 0:
                    maxInd = -cutSlider4.value - 1
                else:
                    maxInd = -1

                rend.glyph.line_alpha = 1
                rend2.glyph.line_alpha = 1
            else:
                rend.glyph.line_alpha = 0
                rend2.glyph.line_alpha = 0

            derivative_values = derivative(val2smooth)
            time_max, max_val, time_min, min_val = getMax(new_frames, new_values, val2smooth.tolist(), threshold)
            source2.data = {'frames': new_frames, 'avgLine': val2smooth}
            source4.data = {'timeMaxima': time_max, 'maxima': max_val,
                            'set': ["auto" for i in range(len(max_val))]}
            tStart, value_start, tEnd, value_end = startEndPeak(new_frames, new_values, derivative_values,
                                                                threshold)
            source5.data = {'timeStart': tStart, 'startValue': value_start,
                            'set': ["auto" for i in range(len(value_start))]}
            source6.data = {'timeEnd': tEnd, 'endValue': value_end,
                            'set': ["auto" for i in range(len(value_end))]}
            source7.data = {'timeMinima': time_min, 'minima': min_val,
                            'set': ["auto" for i in range(len(min_val))]}

            settings.data = {'AvgFiltern': [kernelSlider1.value], 'AvgFilterWidth': [kernelSlider2.value],
                             'SkipInitial': [cutSlider.value], 'SkipLast': [cutSlider2.value],
                             'ThresholdLevel': [thresholdSpinner.value], 'fps': [fpsSpinner.value]}

        def cuttingPoints(attr, old, new, frames, values):
            if cutSlider2.value > 0: maxInd = -cutSlider2.value -1
            else: maxInd = -1
            new_frames = frames[cutSlider.value: maxInd]
            new_values = values[cutSlider.value: maxInd]
            yrange = max(new_values) - min(new_values)
            p1.y_range.start = min(new_values) - yrange / 10
            p1.y_range.end = max(new_values) + yrange / 10
            source1.data = {'frames': new_frames, 'intensity': new_values}

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
            time_max, max_val, time_min, min_val = getMax(new_frames, new_values, val2smooth.tolist(), threshold)

            source2.data = {'frames': new_frames, 'avgLine': val2smooth}
            source4.data = {'timeMaxima': time_max, 'maxima': max_val,
                            'set': ["auto" for i in range(len(max_val))]}
            source5.data = {'timeStart': new_tStart, 'startValue': nvalue_start,
                            'set': ["auto" for i in range(len(nvalue_start))]}
            source6.data = {'timeEnd': new_tEnd, 'endValue': nvalue_end,
                            'set': ["auto" for i in range(len(nvalue_end))]}
            source7.data = {'timeMinima': time_min, 'minima': min_val,
                            'set': ["auto" for i in range(len(min_val))]}

            if cutSlider4.value > 0: maxInd = -cutSlider4.value -1
            else: maxInd = -1
            dy_frames = new_frames[cutSlider3.value: maxInd]
            new_dy = derivative_values[cutSlider3.value: maxInd]

            source3.data = {'frames': dy_frames, 'dy': new_dy}
            settings.data = {'AvgFiltern': [kernelSlider1.value], 'AvgFilterWidth': [kernelSlider2.value],
                             'SkipInitial': [cutSlider.value], 'SkipLast': [cutSlider2.value],
                             'ThresholdLevel': [thresholdSpinner.value], 'fps': [fpsSpinner.value]}

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

        def output_data(attr, old, new):
            if selectDir.value == 'new':
                text_input2.visible = True
                bt3.visible = True
                output.data = {'output_dir': [text_input2.value], 'output_file': [text_input.value],
                               'ext': [ext.labels[ext.active]]}
                settings.data = {'AvgFiltern': [kernelSlider1.value], 'AvgFilterWidth': [kernelSlider2.value],
                                 'SkipInitial': [cutSlider.value], 'SkipLast': [cutSlider2.value],
                                 'ThresholdLevel': [thresholdSpinner.value], 'fps': [fpsSpinner.value]}
            else:
                text_input2.visible = False
                bt3.visible = False
                output.data = {'output_dir': [selectDir.value], 'output_file': [text_input.value],
                               'ext': [ext.labels[ext.active]]}
                settings.data = {'AvgFiltern': [kernelSlider1.value], 'AvgFilterWidth': [kernelSlider2.value],
                                 'SkipInitial': [cutSlider.value], 'SkipLast': [cutSlider2.value],
                                 'ThresholdLevel': [thresholdSpinner.value], 'fps': [fpsSpinner.value]}

        def create_dir():
            outputDir = '../output'
            os.makedirs(os.path.join(outputDir, text_input2.value))
            outputDirs = getOutputDirs('../output')
            selectDir.options = ['new'] + outputDirs
            selectDir.value = text_input2.value
            text_input2.visible = False
            bt3.visible = False

        def callback(attr, old, new, source):
            try:
                # TODO: is there a way to prevent accessing protected member? .remove_on_change not possible with
                #  partial
                if 'value' in valueSlider._callbacks and len(valueSlider._callbacks['value']) == 1:
                    del valueSlider._callbacks['value'][0]
                keys = list(source.data)
                time_key = [i for i in keys if 'time' in i][0]
                data_x = source.data[time_key][new[0]]
                index = frames.index(data_x)
                valueSlider.value = index
                valueSlider.on_change('value', partial(readjust_glyph, frames=frames, values=values,
                                                       point_index=new[0], source=source))
            except IndexError:
                pass

        def readjust_glyph(attr, old, new, source, frames, values, point_index=None):
            keys = list(source.data)
            time_key = [i for i in keys if 'time' in i][0]
            value_key = [i for i in keys if 'Value' in i or 'max' in i][0]
            if point_index is not None:
                source.patch({time_key: [(point_index, frames[valueSlider.value])],
                              value_key: [(point_index, values[valueSlider.value])],
                              "set": [(point_index, 'manual')]})
            else:
                try:
                    new_frame = [i for i in source.data[time_key] if i not in set(frames)][0]
                    closest_frame = round(new_frame)
                    new_value = values[closest_frame]

                    index = source.data[time_key].index(new_frame)
                    source.patch({time_key: [(index, closest_frame)],
                                  value_key: [(index, new_value)],
                                  'set': [(index, 'manual')]})
                except IndexError:
                    pass




        kernelSlider1.on_change('value', partial(updateAvg, frames=frames, values=values))
        kernelSlider2.on_change('value', partial(updateAvg, frames=frames, values=values))
        cutSlider.on_change('value', partial(cuttingPoints, frames=frames, values=values))
        cutSlider2.on_change('value', partial(cuttingPoints, frames=frames, values=values))
        cutSlider3.on_change('value', partial(cuttingDerivative, frames=frames, values=values))
        cutSlider4.on_change('value', partial(cuttingDerivative, frames=frames, values=values))
        text_input.on_change('value', output_data)
        text_input2.on_change('value', output_data)
        selectDir.on_change('value', output_data)
        ext.on_change('active', output_data)
        fpsSpinner.on_change('value', output_data)
        thresholdSpinner.on_change('value', partial(updateAvg, frames=frames, values=values))
        bt.on_click(partial(save, source1, source4, source5, source6, settings, output, p1, p2, df))
        bt2.on_click(partial(noIntervalSave, source1, source4, source7, settings, output, p3, df))
        bt3.on_click(create_dir)
        fileInp2.on_change('value', initialPlot)
        source4.selected.on_change('indices', partial(callback, source=source4))
        source5.selected.on_change('indices', partial(callback, source=source5))
        source6.selected.on_change('indices', partial(callback, source=source6))
        source4.on_change('data', partial(readjust_glyph, source=source4, frames=frames, values=values))
        source5.on_change('data', partial(readjust_glyph, source=source5, frames=frames, values=values))
        source6.on_change('data', partial(readjust_glyph, source=source6, frames=frames, values=values))

        layout2 = row(column(p1, valueSlider, p2), column(kernelSlider1, kernelSlider2, cutSlider, cutSlider2,
                                                          cutSlider3, cutSlider4, thresholdSpinner, fpsSpinner,
                                                          row(text_input, ext, bt), row(selectDir, text_input2, bt3),
                                                          fileInp2))
        layout3 = row(column(p3, valueSlider), column(kernelSlider1, kernelSlider2, cutSlider, cutSlider2,
                                                      thresholdSpinner, fpsSpinner, row(text_input, ext, bt2),
                                                      row(selectDir, text_input2, bt3), fileInp2)
                      )
        intervalPanel = Panel(child=layout2, title="Interval")
        noIntervalPanel = Panel(child=layout3, title="No interval")
        tabs = Tabs(tabs=[intervalPanel, noIntervalPanel])
        doc.clear()
        doc.add_root(tabs)

    fileInp = FileInput(accept=".csv")
    fileInp.on_change('value', initialPlot)
    layout = column(fileInp)
    doc.add_root(layout)


server = Server({'/': start}, num_procs=1)
server.start()

if __name__ == '__main__':
    server.io_loop.add_callback(server.show, "/")
    server.io_loop.start()
