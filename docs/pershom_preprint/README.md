# Persistent Homology as Stopping for Voronoi Interpolation
In this study the Voronoi interpolation is used to interpolate a set of points  drawn from a topological space with higher homology groups on its filtration. The technique is based on Voronoi tesselation, which induces a natural dual map to the Delaunay triangulation. Advantage is taken from this fact calculating the persistent homology on it after each iteration to capture the changing topology of the data. The boundary points are identified as critical. The Bottleneck and Wasserstein distance serve as a measure of quality between the original point set and the interpolation. If the norm of two distances exceeds a heuristically determined threshold, the algorithm terminates. We give the theoretical basis for this approach and justify its validity with numerical experiments.

## References

    @article{lume19,
	  author    = {Luciano Melodia and
	               Richard Lenz},
	  title     = {Persistent Homology as Stopping-Criterion for Voronoi Interpolation},
	  journal   = {CoRR},
	  volume    = {abs/1911.02922},
	  year      = {2019},
	  url       = {http://arxiv.org/abs/1911.02922},
	  archivePrefix = {arXiv},
	  eprint    = {1911.02922}
	}

