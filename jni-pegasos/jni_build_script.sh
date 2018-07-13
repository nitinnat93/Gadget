#!/usr/bin/env bash

cd ./src/pegasos_native
rm -f *.o
make
make install
mv ./src/pegasos_native/libpegasos.so lib
module load ant/1.8.1
ant -f ./src/jnipegasos/build.xml