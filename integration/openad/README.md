# OpenAD/F examples

`run.sh` differentiates examples that ship with the OpenAD/F tool
(snapshot `OpenAD_2014-03-15.tgz`, checked by SHA-256) with Enzyme through
flang, FlangEnzyme's config file and full LTO, and checks each gradient:

| Example | Problem | Check |
|---|---|---|
| BoxModel | Stommel-type three-box ocean model, 6 controls, 3650 time steps | dot product test against central finite differences |
| FDC | MINPACK-2 flow in a driven cavity, 10 x 10 grid | dot product test against central finite differences |
| ShallowWater | channel flow over a sill (Losch and Wunsch 2003), 160 bathymetry controls; the SWM case of Vlasenko et al., Comput. Phys. Commun. 199 (2016) | gradient OpenAD computed (`shallow_water_ad.out`) and finite differences |

The example sources are not modified; each `enzyme_driver.f90` wraps the
example's entry point as `f(n, x, m, y)` for `enzyme_autodiff`.

    ./run.sh 23 build/Enzyme

needs flang-23, ld.lld and netCDF-Fortran, and an Enzyme build with
`-DENZYME_FORTRAN=ON -DENZYME_FLANG=ON` and the `FlangEnzyme-23`,
`LLDEnzyme-23` and `EnzymeFortran` targets. Examples Enzyme cannot handle yet
are listed in `EXPECTED_FAILURES` in `run.sh`.
