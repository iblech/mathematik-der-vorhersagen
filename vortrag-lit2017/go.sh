#!/bin/bash

select foo in \
    "mplayer 'resources/Windows Vista Speech Recognition Tested - Perl Scripting.mp4'" \
    "mplayer 'resources/Hound Internal Demo.mp4'" \
    "mplayer 'resources/speaker-1.wav'" \
    "mplayer 'resources/speaker-2.wav'" \
    "mplayer 'resources/speaker-3.wav'" \
    "chromium-browser 'resources/jcjohnson*.html'" \
    "animate 'resources/neural-enhance.gif'" \
    "vi ../neuronale-netze/ziffern/net-100.p" \
    "python ../neuronale-netze/ziffern/demo.py"
do
    eval "$foo"
done
