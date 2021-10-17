# What is this
This is a quick hack implementation of a simulator for [xkcd 2529's weirdly concrete problem](https://xkcd.com/2529). This is barely tested and I suck at computational geometry, so this implementation is likely not correct.

# How does it work
I generate self-avoiding walks using the dimer method described in (this notebook)[https://github.com/gabsens/SelfAvoidingWalk/blob/master/SAW.ipynb]. There is apparently some more fancy stuff, but I haven't gotten to that.

To find the largest distance line I find the convex hull using a Grahan scan and from that find antipodal points using the rotating caliper technique and take the maximal pair.

To find colinear points I just got lazy and brute forced it.

# How do I run it

There are two entry points:

- `plot_main` which will generate and plot a random walk, its marbles, its largest distance line, and the marbles colinear to that line.
- `stat_main` will generate many random walks, and find the mean and standard deviations of the number of marbles on the line.
