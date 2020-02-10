#!/bin/bash
date1=20180321
date2=20180327
master=/net/jokull/nobak/mzzhong/S1-Evans/track_37/raw_crop/20180321/20180321.raw.slc
slave=/net/jokull/nobak/mzzhong/S1-Evans/track_37/coregSLC/Coarse/20180327/20180327.slc
fixImageXml.py -f -i $master
fixImageXml.py -f -i $slave
ww=256
wh=256
sw=20
sh=20
kw=128
kh=128
mm=50
gross=0
gpuid=0
deramp=0
nwac=5
nwdc=5
outprefix=/net/jokull/nobak/mzzhong/S1-Evans/track_37/cuDenseOffsets/20180321_20180327/cuampcor
runid=20180628
outsuffix=_run_$runid
rm $outprefix$outsuffix*
cuDenseOffsets.py --master $master --slave $slave --ww $ww --wh $wh --sw $sw --sh $sh --mm $mm --kw $kw --kh $kh --gross $gross --outprefix $outprefix --outsuffix $outsuffix --deramp $deramp --gpuid $gpuid --nwac $nwac --nwdc $nwdc
