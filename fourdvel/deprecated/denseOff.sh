#! /bin/bash
isce2gis.py envi -i ../geometry/lat.rdr
isce2gis.py envi -i ../geometry/lon.rdr
#for dir in $1; do
#    if [ -d $dir ];then
#        cd $dir
#        MaskAndFilter.py -d ${dir}_$2.bip -s ${dir}_$2_snr.bip -o . -n 6 -t 0
#        cd ..
#    fi
#done
name=test3dense_ampcor
MaskAndFilter.py -d ${name}.bip -s ${name}_snr.bip -o . -n 6 -t -100000

