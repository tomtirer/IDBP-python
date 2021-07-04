# IDBP-python
Image Restoration by Iterative Denoising and Backward Projections

@article{tirer2018image,\
  &nbsp; &nbsp; title={Image restoration by iterative denoising and backward projections},\
  &nbsp; &nbsp; author={Tirer, Tom and Giryes, Raja},\
  &nbsp; &nbsp; journal={IEEE Transactions on Image Processing},\
  &nbsp; &nbsp; volume={28},\
  &nbsp; &nbsp; number={3},\
  &nbsp; &nbsp; pages={1220--1234},\
  &nbsp; &nbsp; year={2018},\
  &nbsp; &nbsp; publisher={IEEE}\
}

This is a python implementation of IDBP (the original implementation was in matlab: https://github.com/tomtirer/IDBP). \
The code is not optimized for runtime (it mostly uses numpy), in order to facilitate using off-the-shelf denoisers without requiring GPU/PyTorch. \
Still, processing an image often takes just a couple of seconds. \
It builds on the CNN denoisers and some code from github: https://github.com/cszn/DPIR.

@article{zhang2020plug,\
  &nbsp; &nbsp; title={Plug-and-Play Image Restoration with Deep Denoiser Prior},\
  &nbsp; &nbsp; author={Zhang, Kai and Li, Yawei and Zuo, Wangmeng and Zhang, Lei and Van Gool, Luc and Timofte, Radu},\
  &nbsp; &nbsp; journal={arXiv preprint},\
  &nbsp; &nbsp; year={2020}\
}

Modifying the code such that IDBP will use other denoisers and/or handle other ill-posed linear inverse problems should be quite simple.
