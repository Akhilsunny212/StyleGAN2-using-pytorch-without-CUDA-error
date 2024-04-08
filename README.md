# StyleGAN2-using-pytorch-without-CUDA-error
Implementation of StyleGAN2-ada using Pytorch without any version related errors and cuda errors.

<h1>Navigate to Google Colab and start a new notebook.</h1>
<h2>Install Miniconda:</h2>
<p>To manage the specific version dependencies, it's recommended to install Miniconda in your Colab environment. Use the following commands to install Miniconda:</p>
<code>!wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh</code>
<code>!bash Miniconda3-latest-Linux-x86_64.sh -bfp /usr/local</code>
<code>import os</code>
<code>os.environ['PATH'] = "/usr/local/bin:" + os.environ['PATH']</code>
<code>!conda update -n base -c defaults conda -y</code>

<h2>Create a Virtual Environment with Python 3.7:</h2>
<p>With Miniconda installed, create a new virtual environment using Python 3.7:</p>
<code>!conda create -n myenv python=3.7 -y</code>
<p>Activate the newly created environment:</p>
<code>!conda run -n myenv</code>

<h2>Install Dependencies:</h2>
<p>Install PyTorch 1.7.1, torchvision 0.8.2, torchaudio 0.7.2, and cudatoolkit 11.0 in the virtual environment:</p>
<code>!conda install -n myenv pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch -y</code>

<h2>Clone NVIDIA's StyleGAN2-ADA-PyTorch Repository:</h2>
<code>!git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git</code>
<p>Navigate to the cloned repository directory:</p>
<code>%cd stylegan2-ada-pytorch</code>

<h2>Install all the required Libraries:</h2>
<p>Install required libraries such as click, tqdm, requests, psutil, scipy.</p>

<h2>Prepare the Dataset:</h2>
<p>Use the dataset_tool.py script to prepare your dataset for training:</p>
<code>!conda run -n myenv python dataset_tool.py --source /path/to/your/dataset --dest /content/drive/MyDrive/custom_dataset/input_set --width=256 --height=256 --resize-filter=box</code>

<h2>Train the Model:</h2>
<p>Initiate the training process with the train.py script:</p>
<code>!conda run -n myenv python train.py --data /content/drive/MyDrive/custom_dataset/input_set --outdir /content/drive/MyDrive/custom_dataset/results</code>
<p>Use a small kimg number for fast results at low quality. For better quality results, retain the default kimg number. Note that while this will yield higher quality, the training will take longer. If you want to customize the kimg value, update your training command accordingly.</p>
<code>!conda run -n myenv python train.py --data /content/drive/MyDrive/custom_dataset/input_set --outdir /content/drive/MyDrive/custom_dataset/results --kimg=1500</code>

<h2>Generate Images:</h2>
<p>After training, use the generated pickle files with the generator.py script to create images:</p>
<code>!conda run -n myenv python generate.py --outdir=/content/drive/MyDrive/custom_dataset/output --trunc=1 --seeds=85,265,297,849 --network=/content/drive/MyDrive/custom_dataset/result/network-snapshot-000000.pkl</code>
