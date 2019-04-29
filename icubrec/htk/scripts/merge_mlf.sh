#!/bin/bash

echo '#!MLF!#'
for FILE in $@; do
    tail -n +2 $FILE
done
