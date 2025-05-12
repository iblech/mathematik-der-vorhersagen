plot \
  "./training-validation0.log" using 0:1 smooth kdensity bandwidth 10 w l lw 3, \
  "./training-validation0.log" using 0:2 smooth kdensity bandwidth 10 w l lw 3, \
  "./training-validation0.log" using 0:3 smooth kdensity bandwidth 10 w l lw 3, \
  "./training-validation0.log" using 0:4 smooth kdensity bandwidth 10 w l lw 3
pause -1
