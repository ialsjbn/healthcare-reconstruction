#!/bin/bash

mkdir results_6_8
for i in {1..100}; do
    mpirun -n 4 python3 greedy_evaluation_mpi.py 6 8 './results_6_8/' $i;
done

mkdir results_6_10
for i in {1..100}; do
    mpirun -n 4 python3 greedy_evaluation_mpi.py 6 10 './results_6_10/' $i;
done

mkdir results_6_12
for i in {1..100}; do
    mpirun -n 4 python3 greedy_evaluation_mpi.py 6 12 './results_6_12/' $i;
done

mkdir results_6_15
for i in {1..100}; do
    mpirun -n 4 python3 greedy_evaluation_mpi.py 6 15 './results_6_15/' $i;
done

mkdir results_6_20
for i in {1..100}; do
    mpirun -n 4 python3 greedy_evaluation_mpi.py 6 20 './results_6_20/' $i;
done