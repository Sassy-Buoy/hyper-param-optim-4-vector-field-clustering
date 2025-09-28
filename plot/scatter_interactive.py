"""plotting"""

import json
import pathlib
import random
from collections.abc import Callable
from typing import Optional, Union

import plotly.express as ptx
import discretisedfield as df
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objs as go

from IPython.display import display

with open("./data/simulation_file_paths.json", "r", encoding="utf-8") as f:
    simulation_file_paths = json.load(f)
with open("./data/parameters_dict.json", "r", encoding="utf-8") as f:
    parameters_dict = json.load(f)


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
    field.orientation.sel("z").resample(n=(20, 20)).mpl.vector(ax=axs, use_color=False)
    return axs


def plot_all_examples(
    labels,
    parameter_dictionary=parameters_dict,
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
    class_dictionary = c_dict(labels)
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


def c_dict(labels, simulation_file_paths):
    class_dict = {f"Class {lbl}": [] for lbl in set(labels) if lbl != -1}
    class_dict["Outliers"] = []
    image_dict = {f"Class {lbl}": [] for lbl in set(labels) if lbl != -1}
    image_dict["Outliers"] = []

    for index, path in enumerate(simulation_file_paths):
        class_ = labels[index]
        # find the png file with the index
        img_path = f"data/field_images/{index}.png"
        if class_ == -1:
            class_dict["Outliers"].append(str(path))
            image_dict["Outliers"].append(img_path)
        else:
            class_dict[f"Class {class_}"].append(str(path))
            image_dict[f"Class {class_}"].append(img_path)

    return class_dict, image_dict


def scatter_interactive(
    labels, parameters_dict=parameters_dict, simulation_file_paths=simulation_file_paths
):
    """
    Plot the classes in a scatter plot with H on x-axis and E on y-axis.
    On hover, show the field image.
    Args:
        labels : list of int
            List of class labels for each simulation.
        parameters_dict : dict, optional
            Dictionary of parameters for each simulation, by default parameters_dict
        simulation_file_paths : list of str, optional
            List of file paths for each simulation, by default simulation_file_paths
    Returns:
        plotly.graph_objs.FigureWidget: Interactive scatter plot with hover images.
    """
    class_dict, image_dict = c_dict(labels, simulation_file_paths)

    fig = go.FigureWidget()  # FigureWidget allows interactive updates

    # Add scatter traces
    for class_id, colour in zip(class_dict.keys(), ptx.colors.qualitative.Dark24):
        fig.add_scatter(
            x=[parameters_dict[f]["H"] for f in class_dict[class_id]],
            y=[parameters_dict[f]["E"] for f in class_dict[class_id]],
            mode="markers",
            name=class_id,
            marker=dict(size=11, color=colour),
            customdata=image_dict[class_id],
            hovertemplate="H: %{x}<br>E: %{y}<extra></extra>",
        )

    fig.update_layout(
        width=900,
        height=600,
        margin=dict(l=10, r=10, t=25, b=25),
        xaxis_title=r"$\mu_0 \mathbf{H} \text{ (T)}$",
        yaxis_title=r"$\Delta \text{E (J)}$",
        images=[  # initial empty image
            dict(
                source="",
                xref="paper",
                yref="paper",
                x=0.4,  # just outside right of plot
                y=1,
                xanchor="left",
                yanchor="top",
                sizex=0.3,
                sizey=0.3,
                layer="above",
            )
        ],
    )

    # Update image on hover
    def update_image(trace, points, state):
        if points.point_inds:
            idx = points.point_inds[0]
            img_path = trace.customdata[idx]
            fig.layout.images[0].source = img_path

    for trace in fig.data:
        trace.on_hover(update_image)

    display(fig)
