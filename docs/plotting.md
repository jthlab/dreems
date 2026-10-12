# Visualizing Surfaces

Here, we demonstrate several ways to visualize a DREEMS migration surface. Plotting requires only the `surface` object returned by `dreems_infer`. 

We will be using the example fish dataset from the [Allohistium fishes tutorial](./fishes.md)

## Interactive maps
Given a `surface` object, one can create an interactive map that we will save in a file called `fishes.html`.

```python
draw_interactive_map(surface, "./fishes.html")
```

<iframe
    src="_static/fishes.html"
    width="100%"
    height="700"
    style="border: 1px solid #ccc; border-radius: 6px;"
    loading="lazy">
</iframe>

This interactive map allows you to navigate and explore underlying street networks, river systems, parks, and natural landscapes. Blue grid areas represent higher relative effective migration, while red areas indicate lower relative effective migration. Sampled nodes are marked with white circles, scaled proportionally to the sample size at each location. You can customize the base map view by hovering over the layer control in the top-right corner to toggle between `OpenStreetMap` and `Colored terrain` (which highlights elevation).

## Grid maps
We plot the grid map using [`contextily`](https://contextily.readthedocs.io/en/latest/providers_deepdive.html) basemaps and export the output to `figure_path`. The optional pop_map parameter accepts a dictionary that maps sample identifiers to population labels, enabling population-based color coding. You can omit this parameter and all samples are plotted in a single uniform color. Note: if you provide a `figure_path` you must set `save_figure=True`, the default is `save_figure=False`.

```python
draw_projected_grid_map(
    surface,
    sample_to_pop=pop_map,
    save_figure=True,
    figure_path = "./fishes.pdf"
)
```

<iframe
    src="_static/fishes.pdf"
    width="100%"
    height="700px"
    style="border: none;">
</iframe>

## Contour maps
One can also plot contour maps and save it to `figure_path`.

```python
draw_projected_contour_map(
    surface,
    sample_to_pop=pop_map,
    smoothing_km=5.0,
    contour_fill_alpha = 0.30,
    contour_line_alpha=1.0,
    contour_levels=15,
    map_margin_km=75.0,
    posterior=False,
    save_figure=True,
    figure_path = "./fishes_contour.pdf"
)
```

<iframe
    src="_static/fishes_contour.pdf"
    width="100%"
    height="700px"
    style="border: none;">
</iframe>

## Posterior SD plots

In addition to providing point estimates of migration rates, we also have posterior SD (standard deviation) plots that quantify the uncertainty associated with these predictions. You can use the two grid and contour map functions above and simply change `posterior=True`. As one would expect, there's greater uncertainty in unsampled regions as shown below:

```python
draw_projected_contour_map(
    surface,
    sample_to_pop=pop_map,
    smoothing_km=5.0,
    contour_fill_alpha = 0.30,
    contour_line_alpha=1.0,
    contour_levels=15,
    map_margin_km=75.0,
    posterior=True,
    save_figure=True,
    figure_path = "./fishes_posterior_contour.pdf"
)
```

<iframe
    src="_static/fishes_posterior_contour.pdf"
    width="100%"
    height="700px"
    style="border: none;">
</iframe>

## Different contextily maps

Users can provide different `basemap_source` into the grid and contour mapping functions to change the underlying map. To see all existing map options run the following:

```python
import contextily as cx
cx.providers
```

Here is an example of changing the basemap.

```python
from dreems.plotting import draw_projected_contour_map

fig, ax = draw_projected_contour_map(
    surface,
    sample_to_pop=None,
    smoothing_km=5,
    contour_levels=15,
    contour_linewidth=2.0,
    contour_fill_alpha=0.20,
    save_figure=True,
    figure_path="./map_example.pdf",
    posterior=False,
    basemap_source = cx.providers.OpenTopoMap
)
```

<iframe
    src="_static/map_example.pdf"
    width="100%"
    height="700px"
    style="border: none;">
</iframe>