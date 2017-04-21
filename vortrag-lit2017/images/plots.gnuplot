set style line 1 linecolor rgb '#000000' linetype 1 linewidth 2
set style line 2 linecolor rgb '#ff4444' linetype 1 linewidth 4
set style line 3 linecolor rgb '#4444ff' linetype 1 linewidth 4
set style line 4 linecolor rgb '#000000' linetype 2 linewidth 2
set style line 5 linecolor rgb '#4444ff' linetype 2 linewidth 2

unset xtics
unset ytics
unset ztics
unset border
set xzeroaxis ls 1
set yzeroaxis ls 1
set zzeroaxis ls 1
set samples 100000

set terminal pdf size 7cm, 4cm

set xrange [-3:3]
set yrange [-0.1:1.1]

set output "sigmoid.pdf"
set termoption dash
plot 0 t "" w l ls 1, exp(x)/(1+exp(x)) t "" w l ls 3, 1 t "" w l ls 4

set output "cubic-polynomial.pdf"
set xrange [-2:3]
set yrange [-15:15]
plot 0 t "" w l ls 1, x**3 + 3*x**2 - 6*x - 8 t "" w l ls 3

set output "3d-plot.pdf"
set xrange [-5:5]
set yrange [-5:5]
set zrange [-20:70]
set hidden3d
set isosamples 30
set view 62, 105
splot x**2 + y**2 + sin(x)*5 + cos(x-y)*10 t "" w l ls 5
