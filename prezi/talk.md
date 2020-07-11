0. Thank you very much for the opportunity to share my work. This talk presents a topological stop criterion for interpolation techniques. Our initial problem is to augment data to a larger data set without destroying its structure. The term structure refers to topological structure.

1. Consider this set of points. It could be that the points form an annulus. How do we add points without destroying this original shape?

2. The underlying space can be reconstructed using elementary building blocks. It can be triangulated. These are generalized triangles, also called simplices. They are formal sums of k+1 affine independent points in a d-dimensional metric space. The set of the formal sums of the points with coefficients lambda are called simplices. Due to the restrictions that the sum of all lambdas must result in one and lambda itself is greater than or equal to zero, we obtain a convex set.

3. Combining such simplices as sets we obtain a simplicial complex. This is itself a topological space. Each face of a simplex from the simplicial complex K is again in K. Each intersection of two simplices of K is either empty or a common face of both simplices. Intuitively, it means that only points with points, lines with lines and polygons with polygons can be glued together. Left is a valid example, the right one is not valid.

4. Now we come to the monitoring of the shape of the point set. For this we use the homology groups of the simplicial complex we have spanned over the data.

5. Homology groups intuitively count the number of holes in the respective dimension. This means the zeroth homology group counts connected components, the first homology group holes, the second homology group cavities and so on. The formal sums of the simplices become a commutative group with addition.

6. This commutative group is called chain group. It consists of all sub-complexes of the triangulation. These are connected by inclusion. We are studying not only the homology groups of the triangulation, which means we are not only interested in the holes in the largest simplex. We are interested in all holes of all sub-complexes.

7. We investigate them for different parameterizations of the simplicial complex. This means that we study the smaller sub-complexes in the deeper inclusions. The abelian group provides a group homomorphism, the boundary operator. Applying the boundary operator to them, the chains are either mapped to the zero map, in which case the chain is a cycle. Or, if a smaller chain remains, it is a face of another chain.

8. The homology group of a simplicial complex is the quotient of cycles and boundaries. Intuitively speaking, one lets all boundaries collapse, which are cycles. What remains are the cycles that enclose the k dimensional holes. 

9. The rank of this homology group is called the Betti number and indicates the number of k dimensional holes in a topological space. We consider the appearance and disappearance of k dimensional holes along the filtration, which encodes the topological structure of the space and is called persistent homology.

10. Practically, there is an interpolation method that uses complexes, the natural neighbor interpolation.

11. Within this method the Voronoi diagram is constructed by forming the set of points closest to each of the points in the data set, called Voronoi regions.

12. We randomly add a point to the convex hull of the original set of points.

13. Iteratively, the points of the neighboring Voronoi regions are connected to the added point.

14. The perpendicular bisecting this connection defines an area stolen by the new point.

15. These areas define the new Voronoi region of the point. 

16. EMPTY

17. The coordinates of the new point are weighted using these defined Voronoi regions. However, a problem is the implicit truncation to calculate the Voronoi regions. This is an artificial embedding that does not reflect the intrinsic geometry, nor topology of the data.

18. The Voronoi diagram is a complex, but not simplicial. We provide another theoretical result. Dual to the Voronoi diagram is the Delaunay Complex. Ideally, if we could calculate the homology groups of the chain complex on the Delaunay complex, we would not need any additional computations. 

19. We combine our approach with a result from Bauer and Edelsbrunner. The Cech-complex, a simplicial complex which provides the most accurate approximation of the topology of the data space, is said to be homotopy equivalent to the Delaunay complex. If this is the case, we may use the Delaunay complex without restrictions. Indeed, the Cech complex collapses to the Delaunay complex up to simple-homotopy equivalence, preserving homology groups.

20. We use a record of handwritings that contain loops. The points of each handwriting are doubled for each interpolation step. 

21. We randomly insert the points equally distributed into the existing convex hull. Then we apply the interpolation algorithm. The ranks of the homology groups are recorded with their time of origin and time of death in the so-called persistence diagram. Using a hypothesis test based on the Wasserstein metric, we decide for each manuscript in each iteration whether the persistence diagrams correspond to the same distribution. Let me briefly remind you that the latter approximate the topology of the space underlying the data.

22. Elementary statistics show the success. The point in time when new structures can be created (due to constraining to Euclidean space during Voronoi interpolation) can be determined. 

23. In summary, what have we achieved? We have combined the collapse sequence with the Voronoi interpolation. We introduced a mathematical stop criterion on a topological basis and specified a suitable hypothesis test. This has resulted in open questions. How should one deal with points that are not in general position? Which embedding should one choose for the Voronoid diagrams in the interpolation? How can one make this choice? And in which cases of this choice of clipping are the homology groups correctly approximated?