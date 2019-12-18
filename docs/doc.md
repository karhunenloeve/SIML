# pershom_exp_persistence_plots

## retrieve_persistence_distances
```python
retrieve_persistence_distances(path:str, delimiter:str=',', decimals:int=2)
```

Customized retrieval function for the bottleneck distances.
Retrieves elements from a 3-tuple, named by the iteration step it_i.
Works for CSV files only.

:param path: Path of the respective file.
:param delimiter: CSV delimiter, by default set to ','.
:return: A nice message to communicate that we are done.

