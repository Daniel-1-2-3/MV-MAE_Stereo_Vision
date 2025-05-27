<h3 align="left">Model Architecture</h3>

<p align="center">
  <img src="MV-MAE Achitecture.jpg" width="90%" />
</p>

<ol>
  <li>
    <b>Patch Embedding</b><br><br>
    The image is divided into 8×8 patches, which are flattened and passed through a 
    linear projection layer to generate embeddings. Positional embeddings are then added 
    to retain spatial information.  
    <br>
    <i>See:</i> <code>patch_embeddings.py</code>
  </li>
  <br>

  <li>
    <b>Encoder</b><br><br>
    The patch embeddings are passed into the encoder, which incorporates transformer layers located in the 
    <code>TransformerLayer/</code> folder. The encoder consists of: <br><br>
    <ul>
      <li>A <b>multi-head self-attention</b> layer that computes attention vectors for each patch using the softmax function.  
        <br><i>See:</i> <code>multi_head_self_attention.py</code>
      </li>
      <li>A <b>feedforward layer</b> (a fully connected network) that updates each patch vector based on its importance.  
        <br><i>See:</i> <code>feed_fwd.py</code>
      </li>
    </ul>
  </li>
  <br>

  <li>
    <b>Decoder Input Preparation</b><br><br>
    The output from the encoder has its dimensions reduced, preparing it to be fed into a lightweight decoder. 
    Placeholder mask tokens for the masked image are added to the end of the array for patch tokens. 
    Then, positional embeddings are concatenated. The result from this block of procedures will be the input for the decoder.
    <br>
    <i>See:</i> <code>decoder_input_prepare.py</code>
  </li>
  <br>

  <li>
    <b>Decoder</b><br><br>
    The decoder is composed of a multi-head self-attention layer and a feedforward layer, like the encoder. 
    The purpose is to reconstruct the masked image based on the patch tokens generated.
  </li>
</ol>

---

<h3 align="left">Installation</h3>

<ol>
  <li>
    <b>Clone the repository</b><br>
    <pre><code>git clone https://github.com/Daniel-1-2-3/MV-MAE_Stereo_Vision.git
cd MV_MAE_Implementation</code></pre>
  </li>

  <li>
    <b>Install dependencies</b><br>
    <pre><code>pip install torch torchvision Pillow einops numpy opencv-python</code></pre>
  </li>
</ol>

---

<h3 align="left">Demo Instructions</h3>

Run <code>main.py</code> for the following demo:

From the two provided demo images, one image is masked, and the other is used as input to the model. The input is passed through all model layers, including patch embedding, transformer encoding, and decoding, in order to reconstruct the masked image. The reconstructed image is displayed using OpenCV, and an MSE loss is calculated against the original image.

> ⚠️ <b>Note:</b> This is a prototype. Training is not yet implemented.
