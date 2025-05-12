plot \
  "./training-validation0-1.log" using 0:4 smooth kdensity bandwidth 10 w l lw 3, \
  "./training-validation0-2.log" using 0:4 smooth kdensity bandwidth 10 w l lw 3, \
  "./training-validation0-3.log" using 0:4 smooth kdensity bandwidth 10 w l lw 3, \
  "./training-validation0-4.log" using 0:4 smooth kdensity bandwidth 10 w l lw 3
pause -1
