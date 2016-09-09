import json, pandas


def create_scores_file(out_file, **scores):
    with open(out_file, 'w') as f:
        json.dump(dict(_scores = scores),
                  f,
                  indent = 4)


def create_charts_file(out_file, **charts_source):
    with open(out_file, 'w') as f:
        column_names = []
        xs = {}
        columns = []
        
        for chart_title, chart_data in charts_source.viewitems():
            chart_data = pandas.DataFrame(chart_data)
            if chart_data.shape[1] > 0:
                chart_data = pandas.get_dummies(chart_data)
            cur_x_name = '%s_x' % chart_title
            column_names.append(cur_x_name)
            columns.append(list(chart_data.index))
            
            for col in chart_data.columns:
                cur_y_name = '%s_%s' % (chart_title, col)
                column_names.append(cur_y_name)
                columns.append(list(chart_data[col]))
                xs[cur_y_name] = cur_x_name
        
        c3_config = {'columnNames' : column_names,
                     'data' : {'xs' : xs,
                               'columns' : columns},
                     'axis' : {'x' : { 'label' : { 'text' : 'Epochs/Batches/Episodes' } },
                               'y' : { 'label' : { 'text' : 'Metrics' } } },
                     'legend' : { 'position' : 'right' }
                     }
        json.dump(dict(_charts = c3_config),
                  f,
                  indent = 4)
