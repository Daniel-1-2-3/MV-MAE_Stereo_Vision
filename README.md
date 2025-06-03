<h3 align="left">Model Architecture</h3>

<p align="center">
  <img src="MV-MAE Achitecture.jpg" width="90%" />
</p>

<ol>
  <li>
    <b>Patch Embedding</b><br>
    The image is divided into 8×8 patches, which are flattened and passed through a 
    linear projection layer to generate embeddings. Positional embeddings are then added 
    to retain spatial information.  
    <br>
    See: <code>patch_embeddings.py</code>
  </li>
  <br>
  <li>
    <b>Encoder</b><br>
    The patch embeddings are passed into the encoder, which incorporates transformer layers located in the 
    <code>TransformerLayer/</code> folder. The encoder consists of: <br><br>
    <ul>
      <li>A <b>multi-head self-attention</b> layer that computes attention vectors for each patch using the softmax function.  
        <br>See: <code>multi_head_self_attention.py</code>
      </li>
      <li>A <b>feedforward layer</b> (a fully connected network) that updates each patch vector based on its importance.  
        <br>See: <code>feed_fwd.py</code>
      </li>
    </ul>
  </li>
  <br>
  <li>
    <b>Decoder Input Preparation</b><br>
    The output from the encoder has its dimensions reduced, preparing it to be fed into a lightweight decoder. 
    Placeholder mask tokens for the masked image are added to the end of the array for patch tokens. 
    Then, positional embeddings are concatenated. The result from this block of procedures will be the input for the decoder.
    <br>See: <code>decoder_input_prepare.py</code>
  </li>
  <br>
  <li>
    <b>Decoder</b><br>
    The decoder is composed of a multi-head self-attention layer and a feedforward layer, like the encoder. 
    The purpose is to reconstruct the masked image based on the patch tokens generated.
  </li>
  <br>The full model is defined in <code>model.py</code>
</ol>

<h3 align="left">General Installation</h3>
<ol>
  <li>
    <b>Clone the repository</b><br>
    <pre><code>git clone https://github.com/Daniel-1-2-3/MV-MAE_Stereo_Vision.git
cd MV_MAE_Implementation</code></pre>
  </li>
  <li>
    <b>Install dependencies</b><br>
    <pre><code>pip install torch torchvision Pillow einops numpy opencv-python pybullet hashlib tqdm</code></pre>
  </li>
</ol>

<h3 align="left">Simulated Stereo Vision for Dataset Image Collection</h3>
<ol>
  <li>
    <b>Run Simulation</b><br>
    Run the following file: <code>build_dataset.py</code>. This script launches a PyBullet simulation with a red cube and
    blue sphere, and a pair of parallel stereo cameras that orbit around the objects. The program will change zoom, yaw and pitch
    of the camera to orbit the objects, capturing all views:
    <br>
    <br>
    Left and right views will be saved to the <code>Dataset</code> folder, with a 70% vs 30% split between the training and validation sets:
    <pre><code>
      Dataset/      
        ├── Train/
        │   ├── LeftCam/
        │   └── RightCam/
        └── Val/
            ├── LeftCam/
            └── RightCam/
    </pre></code>
  </li>
</ol>
      
<h3 align="left">Run Training</h3>
<ol>
  <li>
    Run the file <code>train.py</code>. An example command is shown below:
    <pre><code>python train.py --img_size 256 --patch_size 8 --batch_size 32 --num_epochs 100 --lr 0.0004</code></pre>
  </li>
  <li>
    The following parameters are customizable:<br><br>
    <ul>
      <li><code>--img_size</code>: (default: 256)</li>
      <li><code>--patch_size</code>: (default: 8)</li>
      <li><code>--batch_size</code>: (default: 32)</li>
      <li><code>--in_channels</code>: (default: 3)</li>
      <li><code>--encoder_embed_dim</code>: dimension of each patch embedding in encoder (default: 768)</li>
      <li><code>--encoder_num_heads</code>: number of heads in encoder attention layer (default: 12)</li>
      <li><code>--decoder_embed_dim</code>: dimension of each patch embedding in decoder (default: 512)</li>
      <li><code>--decoder_num_heads</code>: number of heads in decoder attention layer (default: 8)</li>
      <li><code>--num_epochs</code>: (default: 100)</li>
      <li><code>--lr</code>: (default: 0.0004)</li>
    </ul>
  </li><br>
  <li>
    Weights are saved every 50 epochs during training, to a folder named <code>/Weights</code>. 
    After training, to perform inference with the saved weights, run <code>model.py</code> using the command<br><br>
    <pre><code>python model.py --weights [your_weight_file].pth</code></pre>
    The program automaticaly selects a pair of images from the dataset and attempts to perform reconstruction. The reconstructed and ground truth images will be
    displayed, along with the MSE loss.
  </li>
</ol>
