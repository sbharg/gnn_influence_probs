#!/bin/sh

eps=0.03
./imm_discrete -dataset ego-facebook-gnn/ -k 1 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-gnn/ -k 2 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-gnn/ -k 4 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-gnn/ -k 8 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-gnn/ -k 16 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-gnn/ -k 32 -model IC -epsilon $eps

./imm_discrete -dataset ego-facebook-slicer/ -k 1 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-slicer/ -k 2 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-slicer/ -k 4 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-slicer/ -k 8 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-slicer/ -k 16 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-slicer/ -k 32 -model IC -epsilon $eps

./imm_discrete -dataset ego-facebook-actual/ -k 1 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-actual/ -k 2 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-actual/ -k 4 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-actual/ -k 8 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-actual/ -k 16 -model IC -epsilon $eps
./imm_discrete -dataset ego-facebook-actual/ -k 32 -model IC -epsilon $eps