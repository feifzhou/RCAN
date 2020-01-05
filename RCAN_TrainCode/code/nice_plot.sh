pics=(6393 6400 6406)
npics=${#pics[@]}
for i in ${pics[@]}; do
  convert -scale 400% ${i}_x4_LR.png tmp_near_${i}.png
  convert -resize 400% -filter Catrom ${i}_x4_LR.png tmp_bicubic_${i}.png
  convert -resize 400% -filter Gaussian ${i}_x4_LR.png tmp_gauss_${i}.png
done
n1=$((npics-1))
images=$(
for i in `seq 0 $n1`; do
  j=${pics[$i]}
  #if [ $i == $n1 ]; then echo "-label Low_Resolution"; fi
  #echo ${j}_x4_LR.png
  #if [ $i == $n1 ]; then echo "-label Nearest"; fi
  #echo tmp_near_${j}.png
  #if [ $i == $n1 ]; then echo "-label Gaussian"; fi
  #echo tmp_gauss_${j}.png
  if [ $i == $n1 ]; then echo "-label Low_Res(Bicubic)"; fi
  echo tmp_bicubic_${j}.png
  if [ $i == $n1 ]; then echo "-label Neural_network"; fi
  echo ${j}_x4_SR.png
  if [ $i == $n1 ]; then echo "-label High_Resolution"; fi
  echo ${j}_x4_HR.png
done
)
montage -pointsize 36 -frame 4 -mattecolor SkyBlue -geometry '1x1+0+0<' -tile "x${#pics[@]}" $images nice_plot.png
