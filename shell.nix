{ pkgs ? import <nixpkgs> {} }:

let
  zig = pkgs.zig;
in
pkgs.mkShell {
  buildInputs = with pkgs; [
    (python3.withPackages (ps: with ps; [
      numpy
      safetensors
      torch
      torchvision
    ]))
    stdenv.cc.cc.lib
    zig
  ];

  shellHook = ''
    export LD_LIBRARY_PATH=${pkgs.stdenv.cc.cc.lib}/lib:$LD_LIBRARY_PATH
  '';
}

