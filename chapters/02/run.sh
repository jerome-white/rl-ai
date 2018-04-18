#!/bin/bash

epsilon=(
    0
    0.01
    0.1
)

python exercise-2.2.py \
       --epsilon `sed -e's/ / --epsilon /g' <<< ${epsilon[@]}` \
       --bandits 2000 \
       --arms 10 \
       --pulls 1000 |
    python figure-2.1.py --output figure-2.1.png
