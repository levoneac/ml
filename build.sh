#!/bin/sh

set -xe

if [ -f ./outputs/main ]
    then rm ./outputs/*
fi


cc -std=c2x -Wall -Wextra -pedantic -o outputs/main main.c matrix.c neural_net.c read_csv.c -lm -O3 | less -N -X
#cc -std=c2x -Wall -Wextra -o outputs/read_csv read_csv.c | less -N -X
#cc -std=c2x -Wall -Wextra -o outputs/csv main.c | less -N -X

outputs/main > model5.txt
#outputs/read_csv | less -N -X
#outputs/csv | less -N -X
