sed "s/f2c_\(.*\)(/\1_EXTRAWIDTH(/g" -i BLAS/WRAP/cblaswr.c
echo "#include <stdint.h>" >> BLAS/WRAP/stage.c
cat INCLUDE/f2c.h >> BLAS/WRAP/stage.c
cat BLAS/WRAP/cblaswr.c >> BLAS/WRAP/stage.c

sed "s/#include \"f2c.h\"//g" -i BLAS/WRAP/stage.c

sed "s/typedef long int integer;/typedef int32_t integer;/g" BLAS/WRAP/stage.c | sed "s/EXTRAWIDTH//g" > BLAS/WRAP/bclib32.c
sed "s/typedef long int integer;/typedef int64_t integer;/g" BLAS/WRAP/stage.c | sed "s/EXTRAWIDTH/64_/g" > BLAS/WRAP/bclib64.c
