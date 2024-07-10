""" plotting
"""

import pathlib
import random
from collections.abc import Callable
from typing import Optional, Union

import discretisedfield as df
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
# import plotly.express as px
# from dash import Dash, Input, Output, dcc, html
# from dash.exceptions import PreventUpdate
from numpy.typing import NDArray


def path_as_title(
    filepath: str,
    parameter_dictionary: Optional[dict[str, dict[str, float]]],
) -> str:
    """Extracts `sim_name/drive-n` from filepath.

    Parameters
    ----------
    filepath : str

            Filepath to field file.

    parameter_dictionary : dict[str, dict[str, float]]], optional

            Dictionary of pairs of sample keys and parameter dictionaries


    Returns
    -------
        str
            `sim_name/drive-n`.

    """
    return "/".join(pathlib.Path(filepath).parts[-3:-1])


def plot_one_field(axs: plt.Axes, field: df.Field) -> plt.Axes:
    """Default plotting function for example plots.

    Plots the normalised field as a combined scalar/vector plot
    with fixed resampling to (20, 20) for the vector layer.

    Parameters
    ----------
    axis : plt.Axes

        Axis to plot on.

    field : df.Field

        Field to be plotted.

    Returns
    -------
    axis : plt.Axes
        axis with example plot.
    """
    field.orientation.z.sel("z").mpl.scalar(ax=axs, clim=(-1, 1))
    field.orientation.sel("z").resample(
        n=(20, 20)).mpl.vector(ax=axs, use_color=False)
    return axs


def plot_all_examples(
    class_dictionary: dict[str, list[str]],
    parameter_dictionary: Optional[dict[str, dict[str, float]]] = None,
    data_dictionary: Optional[dict[str, df.Field]] = None,
    output_dir: Union[pathlib.Path, str] = ".",
    plot_field: Callable[[plt.Axes, df.Field], plt.Axes] = plot_one_field,
    title_builder: Callable[
        [str, Optional[dict[str, dict[str, float]]]], str
    ] = path_as_title,
) -> None:
    """Generate plots containing samples for each of the classes
    in the provided dictionary.

    Will save pdf files containing visualizations for 9 or less samples
    depending on class size. If more than 9 are available, samples are
    randomly selected. A custom plotting method may be provided.

    Parameters
    ----------
    class_dictionary: dict[str, list[str]]

        Dictionary of pairs of class ids and lists of sample keys.

    parameter_dictionary: dict[str, dict[str, float]], optional

        Dictionary of pairs of sample keys and dictionary
        of pairs of parameter name and parameter value.

    data_dictionary: dict[str, df.Field], optional

        Dictionary of pairs of sample keys and df.Field objects.
        Should be used if fields to be plotted aren't being read from disk.

    plot_field: Callable[[plt.Axes, df.Field, str, str], plt.Axes],
                default: plot_one_field

        Plotting method to visualize sampled fields from each class.
        See `plot_one_field` for default behaviour.

    output_dir: Union[str, pathlib.Path], optional, default `.`

        Path to directory under which to store the output files.
        Defaults to working directory.

    title_builder : Callable[[str, Optional[dict[str, dict[str, float]]]], str]
                    default: path_as_title

        Function to build titles for examples plots.
        See `path_as_title` for default behaviour.
    """
    for key in class_dictionary.keys():
        plot_class_examples(
            class_dictionary=class_dictionary,
            parameter_dictionary=parameter_dictionary,
            key=key,
            data_dictionary=data_dictionary,
            plot_field=plot_field,
            title_builder=title_builder,
        )


def plot_class_examples(
    class_dictionary: dict[str, list[str]],
    key: str,
    parameter_dictionary: Optional[dict[str, dict[str, float]]] = None,
    data_dictionary: Optional[dict[str, df.Field]] = None,
    figsize: tuple[float, float] = (12, 12),
    plot_field: Callable[[plt.Axes, df.Field], plt.Axes] = plot_one_field,
    title_builder: Callable[
        [str, Optional[dict[str, dict[str, float]]]], str
    ] = path_as_title,
    filename: Union[str, pathlib.Path] = None,
) -> matplotlib.figure.Figure:
    """Generate a figure containing visualizations for 9 or less samples of
    the desired class.

    Will randomly sample if more than 9 samples are available. Can select
    which magnetization component to visualize.

    Parameters
    ----------
    class_dictionary : dict[str, list[str]]

        Dictionary of pairs of class ids and lists of sample keys.

    parameter_dictionary : dict[str, dict[str, float]], optional

        Dictionary of pairs of sample keys and dictionary
        of pairs of parameter name and parameter value.

    key : str

        Key for dictionary indicating which class to plot.

    data_dictionary: dict[str, df.Field], optional

        Dictionary of pairs of sample keys and df.Field objects.
        Should be used if fields to be plotted aren't being read from disk.

    figsize: tuple[float, float], default (12,12)

        Size of figure for example plots.

    plot_field: Callable[[plt.Axes, df.Field], plt.Axes], default plot_one_field

        Plotting method to visualize sampled fields from each class.
        See `plot_one_field` for default behaviour.

    title_builder : Callable[[str, Optional[dict[str, dict[str, float]]]], str]
                    default path_as_title

        Function to build titles for examples plots.
        See `path_as_title` for default behaviour.

    filename: Union[str, pathlib.Path], optional

        Path under which to store the output. Filename suffix
        dictates fileformat, see matplotlib documentation for details
        on available formats.

    Returns
    -------
    fig: Figure

        Figure containing the plots of samples belonging to the
        desired class.

    """
    if key not in class_dictionary.keys():
        raise KeyError(f"Invalid class {key} provided.")
    examples = class_dictionary[key]
    random.shuffle(examples)

    fig, axs = plt.subplots(figsize=figsize, nrows=3, ncols=3)
    fig.suptitle(f"{key} - {len(examples)} samples")

    for j in range(min(len(examples), 9)):
        if data_dictionary is None:
            field = df.Field.from_file(examples[j])
        else:
            field = data_dictionary[examples[j]]
        plot_field(axs.flat[j], field)
        axs.flat[j].set_title(title_builder(examples[j], parameter_dictionary))
    fig.tight_layout()

    if filename is not None:
        fig.savefig(filename)

    return fig

# def interactive_2d_scatter_plotly(
#     parameter_dictionary: dict[str, dict[str, float]],
#     x_parameter: str,
#     y_parameter: str,
#     class_dictionary: dict[str, list[str]],
#     opacity: float = 0.7,
#     marker_size: int = 20,
#     c_list: list[str] = px.colors.qualitative.Plotly,
#     **kwargs,
# ):

#     # Initialize the Dash app
#     app = Dash(__name__)

#     traces = []
#     for class_id, colour in zip(class_dictionary, c_list):
#         traces.append(
#             go.Scatter(
#                 x=[
#                     parameter_dictionary[f][x_parameter]
#                     for f in class_dictionary[class_id]
#                 ],
#                 y=[
#                     parameter_dictionary[f][y_parameter]
#                     for f in class_dictionary[class_id]
#                 ],
#                 customdata=[f for f in class_dictionary[class_id]],
#                 mode="markers",
#                 name=class_id,
#                 opacity=opacity,
#                 marker=dict(
#                     size=marker_size,
#                     color=colour,
#                 ),
#                 **kwargs,
#             )
#         )

#     layout = go.Layout(
#         clickmode="event+select",
#         xaxis=dict(title=x_parameter, mirror=True, showline=True, zeroline=False),
#         yaxis=dict(title=y_parameter, mirror=True, showline=True, zeroline=False),
#     )

#     def blank_fig():
#         fig = go.Figure(go.Scatter(x=[], y=[]))
#         fig.update_layout(template=None)
#         fig.update_xaxes(showgrid=False, showticklabels=False, zeroline=False)
#         fig.update_yaxes(showgrid=False, showticklabels=False, zeroline=False)

#         return fig

#     # Define the app layout
#     app.layout = html.Div(
#         children=[
#             dcc.Graph(
#                 id="graph_interaction",
#                 figure={
#                     "data": traces,
#                     "layout": layout,
#                 },
#                 style={"display": "inline-block", "width": "50%"},
#             ),
#             dcc.Graph(
#                 id="config",
#                 style={"display": "inline-block", "width": "25%"},
#                 figure=blank_fig(),
#             ),
#             dcc.Graph(
#                 id="init",
#                 style={"display": "inline-block", "width": "25%"},
#                 figure=blank_fig(),
#             ),
#         ]
#     )

#     # Define callback to update image src on hover
#     @app.callback(Output("config", "figure"), Input("graph_interaction", "hoverData"))
#     def display_hover_data_config(hoverData):
#         if hoverData:
#             field = df.Field.from_file(hoverData["points"][0]["customdata"])
#             fig_s = px.imshow(
#                 np.transpose(field.orientation.z.sel("z").array.squeeze()),
#                 color_continuous_scale="RdBu_r",
#                 zmin=-1,
#                 zmax=1,
#                 origin="lower",
#             )
#             fig_s.update(
#                     layout_coloraxis_showscale=False,
#                     layout_margin=dict(l=20, r=20, t=20, b=20),
#                     )
#             return fig_s
#         else:
#             raise PreventUpdate

#     # Define callback to update image src on hover
#     @app.callback(Output("init", "figure"), Input("graph_interaction", "hoverData"))
#     def display_hover_data_init(hoverData):
#         if hoverData:
#             path = pathlib.Path(hoverData["points"][0]["customdata"]).parent/"m0.omf"
#             field = df.Field.from_file(path)
#             fig_s = px.imshow(
#                 np.transpose(field.orientation.z.sel("z").array.squeeze()),
#                 color_continuous_scale="RdBu_r",
#                 zmin=-1,
#                 zmax=1,
#                 origin="lower",
#             )
#             fig_s.update(
#                     layout_coloraxis_showscale=False,
#                     layout_margin=dict(l=20, r=20, t=20, b=20),
#                     )
#             return fig_s
#         else:
#             raise PreventUpdate


#     return app.run_server(debug=True)
