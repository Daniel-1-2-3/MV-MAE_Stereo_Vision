<h3 align="left">Model Architecture</h3>
<p align="left">
  <img src="MV-MAE Achitecture.jpg" width="100%"/>
</p>
<td>
  Model description and what files represent what layers will be added soon.
</td>

<h3 align="left">Installation</h3>
<ol>
  <li>
    <b>Clone the repository</b>
    
    git clone https://github.com/Daniel-1-2-3/MV-MAE_Stereo_Vision.git
    cd MV_MAE_Implementation
  </li>
  
  <li>
    <b>Install dependencies</b>

    pip install torch torchvision Pillow einops numpy opencv-python
  </li>
</ol>

<h3 align="left">Demo Instructions</h3>

Run <code>main.py</code> for the following demo:

From the two provided demo images, one image is masked, and the other is used as input to the MV-MAE model. The input is passed through all model layers, including patch embedding, transformer encoding, and decoding, in order to reconstruct the masked image. The reconstructed image is displayed using cv2, and an MSE loss is calculated against the original image.

> ! This is a prototype implementation. Training is not yet implemented.

