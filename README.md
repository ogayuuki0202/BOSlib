# BOSlib
# It's still in development so don't use this code!
BOSlib is a library developed to provide open access to various methods of the Background Oriented Schlieren (BOS) method. We are also developing software that runs on a GUI for those who do not have Python skills.

## Key Features
- Short, concise code for visualization, 3D reconstruction, and quantification
- GPU parallel processing is also [available](https://github.com/ogayuuki0202/BOSlib-GPU)

## Warning

The BOSlib is still in its *beta* state. This means that
it still might have some bugs and the API may change. However, testing and contributing
is very welcome, especially if you can contribute with new algorithms and features.

## Installing
Use PyPI: <https://pypi.python.org/pypi/BOSlib>:

    pip install BOSlib 

Or compile from source

Download the package from the Github: https://github.com/ogayuuki0202/BOSlib/archive/refs/heads/main.zip
or clone using git

    git clone https://github.com/ogayuuki0202/BOSlib.git
    cd BOSlib
    python setup.py install 


## Methods

Please see our [Documentation](https://ogayuuki0202.github.io/BOSlib/index.html).

## Getting Started
Here's a quick example of using BOSlib for flow visualization:
1. [3D quantitative visualization and measurement using Abel transform](https://colab.research.google.com/drive/1-Z0ufw8g7u86d0KtyjZTSHDbtDhhknmj?usp=sharing)
2. [3D quantitative visualization and measurement using ARTmethod(CT)]()

## Contributors
- [Yuuki Ogasawara](https://orcid.org/0009-0004-0350-2185)
- Ayumu Ishibashi 
- Narumi Nose
- [Shinsuke Udagawa](https://www.researchgate.net/profile/Shinsuke-Udagawa)
## How to cite this work
If you find this project useful, please cite:

    Yuuki Ogasawara, Ayumu Ishibashi, Shinsuke Udagawa. BOSlib:Background oriented shlieren methods in Python. https://github.com/ogayuuki0202/BOSlib

## How to Contribute
We welcome contributions! If you’d like to report a bug or request a feature, please open an issue on our [GitHub Issues page](https://github.com/ogayuuki0202/BOSlib/issues). We also encourage pull requests for new algorithms and improvements to the library.
