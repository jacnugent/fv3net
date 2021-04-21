let
  pkgs = import (builtins.fetchGit {
    url = "https://github.com/VulcanClimateModeling/fv3gfs-fortran";
    ref = "master";
    rev = "e42cd14e08d6b08910fdec7912a3cc4377145e8a";
  });
  
in 
with pkgs;
mkShell {
  venvDir = "./.venv";
  buildInputs = [
    python3Packages.venvShellHook
    jq
    google-cloud-sdk
    graphviz

    # wrapper inputs
    python3Packages.jinja2
    fms
    esmf
    nceplibs
    netcdf
    netcdffortran
    lapack
    blas
    # this is key: https://discourse.nixos.org/t/building-shared-libraries-with-fortran/11876/2
    gfortran.cc.lib
    gfortran
    cython
    fv3
    mpich
    pkg-config
  ];

  postShellHook = ''
    export MPI=mpich
  '';
}