A little project I did trying to learn C following the concpets given in [Tsoding's ml playlist](https://www.youtube.com/playlist?list=PLpM-Dvs8t0VZPZKggcql-MmjaBdZKeDMw).

Currently the model works by using the finite difference formula to optimize each parameter. A usage example can be found in main.c

# Build
On windows you could run:
```
gcc -o output/main main.c matrix.c neural_net.c read_csv.c -lm
```
With your compiler of choice, and then run the exe.
On linux you could do the same or run build.sh

