# MADFF-GAN: Transmission Line Low-Light Image Enhancement Based on Multi-Directional Attention and Discrepancy Feature Fusion
Xingyao Huang, Yuxiang Wu


[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC_BY--NC_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Code License: Apache 2.0](https://img.shields.io/badge/Code_License-Apache_2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

### Training process

run the following command

```python scripts/script.py --train```

### Testing process

Create directories `../test_dataset`. Put your test images on `../test_dataset`.

Run

```python scripts/script.py --predict ```

### Dataset preparing



If you find this work useful for you, please cite
```
@article{MADFFGAN2026,
  title={Transmission Line Low-Light Image Enhancement Based on Multi-Directional Attention and Discrepancy Feature Fusion},
  author={Huang, Xingyao and Wu, Yuxiang},
  journal={Journal of Visual Communication and Image Representation},
  year={2026},
  note={Accepted. To appear.}
}
```

## License

### Documentation and Data
All non-code materials in this repository, including the manuscript, figures, and datasets,
are licensed under the Creative Commons Attribution-NonCommercial 4.0 International License
(see [LICENSE](LICENSE)).

### Original Source Code
All source code that I have authored (i.e., files not covered by third-party licenses)
is licensed under the Apache License, Version 2.0 (see [LICENSE-CODE](LICENSE-CODE)).
You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

### Third-Party Code
Portions of this project use or adapt code from:

- **EnlightenGAN**, Copyright (c) 2019 Yifan Jiang and Zhangyang Wang (BSD License)
- **DCGAN (dcgan.torch)**, Copyright (c) 2015 Facebook, Inc. (BSD License)

The full license texts for these components are available in the
[third_party_licenses/](third_party_licenses/) directory. The required attributions
are also listed in the [NOTICE](NOTICE) file, as mandated by the Apache License 2.0.
