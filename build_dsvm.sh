#!/bin/bash

echo building the pegasos-native code

module load gcc/7.3.0
rm -rf ./jni-pegasos/src/pegasos-native/*.o
make -C ./jni-pegasos/src/pegasos-native/
make -C ./jni-pegasos/src/pegasos-native/ install

echo building JNI-Pegasos
module load ant/1.8.1
ant -f ./jni-pegasos/src/jnipegasos/build.xml

# move JNI and Pegasos libraries into the peersim-pegasos folder
mv ./jni-pegasos/lib/jnipegasos.jar ./peersim-pegasos/lib
mv ./jni-pegasos/lib/libpegasos.so ./peersim-pegasos/lib

echo building peersim
make -C ./peersim-pegasos/

echo "Peersim successfully built"

