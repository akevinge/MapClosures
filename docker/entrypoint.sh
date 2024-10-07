#!/bin/bash

if [ -z ${@+x} ]; then
    exec bash
else 
    exec bash -c "$@"
fi