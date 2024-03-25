#!/bin/sh

set -xe

if [ -f ./outputs/main ]
    then rm ./outputs/*
fi


cc -std=c2x -Wall -Wextra -o outputs/main main.c matrix.c neural_net.c -lm -Og | less -N -X


outputs/main | less -N -X
