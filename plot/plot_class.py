""" plot_class
"""

import json
import plotly.graph_objs as go
import plotly.express as ptx

# load the parameters_dict.json
with open("data/parameters_dict.json", "r") as f:
    parameters_dict = json.load(f)


def plot_class(class_dict):
    """ plot_class """
    traces = list()
    for class_id, colour in zip(class_dict, ptx.colors.qualitative.Dark24):
        traces.append(
            go.Scatter(
                x=[
                    parameters_dict[f]["H"]
                    for f in class_dict[class_id]
                ],
                y=[
                    parameters_dict[f]["E"]
                    for f in class_dict[class_id]
                ],
                mode="markers",
                name=class_id,
                opacity=0.7,
                marker=dict(
                    size=11,
                    color=colour,
                ),
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        autosize=False,
        width=990,
        height=600,
        margin=dict(
            l=10,
            r=10,
            b=25,
            t=25,
        ),
        xaxis_title=r"$\mu_0 \mathbf{H} \text{ (T)}$",
        yaxis_title=r"$\Delta \text{E (J)}$",
    )

    fig.show()
