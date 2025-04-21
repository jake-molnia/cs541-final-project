{
  description = "dev-env";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config = {
            allowUnfree = true;
          };
        };
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            pkgs.just
            pkgs.uv
            pkgs.python3
          ];

          shellHook = ''
            echo "Ready!"
            echo "Python: $(python --version)"
            echo "uv: $(uv --version)"
            echo "just: $(just --version)"
            echo ""
            echo "To install packages: uv pip install -r requirements.txt"
          '';
        };
      }
    );
}
