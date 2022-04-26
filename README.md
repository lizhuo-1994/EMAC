
## 1 Miniconda

  * wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  * chmod +x Miniconda3-latest-Linux-x86_64.sh
  * ./Miniconda3-latest-Linux-x86_64.sh
  * echo 'export PATH="$pathToMiniconda/anaconda3/bin:$PATH"' >> ~/.bashrc
  * source ~/.bashrc
  * (optional) conda config --set auto_activate_base false

## 2 Install & activate environment:  
  download [drl.tar.gz](https://drive.google.com/file/d/1rNu1hupPVQCL0Cp460tiCVXA55ZkbnH2/view?usp=sharing)

  * tar -zxzf atari_env.tar.gz 
  * mv drl ~/conda/envs/
  * conda activate drl

## 3 Install Mujoco:

  * download [mujoco.tar.gz](https://drive.google.com/file/d/1Pi3HWx5ZPe92WxtJ8lEZzI3tPBQ15J8-/view?usp=sharing)
  * tar -zxzf mujoco.tar.gz 
  * mv .mujoco /home/YOURACCOUNT/
  * echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/YOURACCOUNT/.mujoco/mujoco210/bin' >> ~/.bashrc
  * echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia' >> ~/.bashrc
  * source ~/.bashrc

## 4 Execution:

  * bash scripts/InvertedPendulum-v2/train.sh

## 5 Results

   In result_rewards/InvertedPendulum-v2

  
  
