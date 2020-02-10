#!/bin/bash
offsetfile=/net/jokull/nobak/mzzhong/S1-Evans/track_37/cuDenseOffsets/20180321_20180327/cuampcor_run_20180703.bip
snrfile=/net/jokull/nobak/mzzhong/S1-Evans/track_37/cuDenseOffsets/20180321_20180327/cuampcor_run_20180703_snr.bip
snr_threshold=6
filter_winsize=8
output_dir=/net/jokull/nobak/mzzhong/S1-Evans/track_37/cuDenseOffsets/20180321_20180327
MaskAndFilter.py -d $offsetfile -s $snrfile -n $filter_winsize -t $snr_threshold -o $output_dir
