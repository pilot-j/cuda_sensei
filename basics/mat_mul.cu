/*Basic Idea:
threads represent a running variable that essentially is like
'i' or loop counter in for/while loops
For 1st implementation lets assume we have m threads.

But why m?
for p from 1 to m
    A[i][p]*B[p][j] = OUT[i][j]

So, for basic expression we have 1 for loop that would run m times so m threads.

However, since i, j are independent but different (i is row, j is col)
we might need separate loop variables to populate them. One way around this is:

Imagine a linear map of matA as a collection of N items each with M entries.
Similarly, matB (or technically transpose of matB) can be
thought as linear map of R items each with M entries.

However, the probelm is still not parallelizable as the collections are not of same size
so we will have to write for loops still.

Another view is having 2 collections each of M size with entry size as N and R respectively.
.
*/
