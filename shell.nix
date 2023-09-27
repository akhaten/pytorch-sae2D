{ pkgs ? import <nixpkgs> {}, ... }:

pkgs.mkShell {

    buildInputs = with pkgs; [

        python310
        python310Packages.torchWithoutCuda
        python310Packages.ignite
        python310Packages.torchvision
        python310Packages.scikitimage
        python310Packages.scikit-learn
        python310Packages.matplotlib
        python310Packages.numpy
        python310Packages.tqdm


        # Package for Jupyter / To comment
        python310Packages.ipywidgets
        python310Packages.ipykernel
        python310Packages.ipympl
        python310Packages.ipython
        
        
        
        

        

   

        

    ];

}
