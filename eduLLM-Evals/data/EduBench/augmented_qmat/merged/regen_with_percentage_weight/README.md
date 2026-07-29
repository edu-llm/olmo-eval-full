This uses a different metric criteria for edge weighting, where we consider how "important" this edge is to each bounding node.

Let Skill A and Skill B be connected via edge. Using the unmerged, but augmented q-map, we first take the number of criterion shared between both A and B (call this V). Then we divide V by the total number of criterion "correlated with" A and then separately divide V by the total number of criterion "correlated with" B. 


Then we take the minimum of these two values to define the edge weight. The idea is that if one of the values is low, then the overall node weight should be low, since this doesn't represent super robust connection.