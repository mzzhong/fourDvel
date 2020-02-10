#!/bin/bash
date1=20180321
date2=20180327
master=/net/jokull/nobak/mzzhong/S1-Evans/track_37/merged/SLC/20180321/20180321.slc.full
slave=/net/jokull/nobak/mzzhong/S1-Evans/track_37/merged/SLC/20180327/20180327.slc.full
fixImageXml.py -f -i $master
fixImageXml.py -f -i $slave
ww=256
wh=128
sw=20
sh=20
kw=128
kh=64
mm=50
gross=0
gpuid=5
deramp=0
nwac=10
nwdc=10
outprefix=/net/jokull/nobak/mzzhong/S1-Evans/track_37/cuDenseOffsets/20180321_20180327/cuampcor
runid=20180703_dev
outsuffix=_run_$runid
cuDenseOffsets.py --master $master --slave $slave --ww $ww --wh $wh --sw $sw --sh $sh --mm $mm --kw $kw --kh $kh --gross $gross --outprefix $outprefix --outsuffix $outsuffix --deramp $deramp --gpuid $gpuid --nwac $nwac --nwdc $nwdc
