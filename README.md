## Using State Predictions for Value Regularization in Curiosity Driven Deep Reinforcement Learning ##

### About
Tensorflow implementation of the algorithm described in ‘Using State Predictions for Value Regularization in Curiosity Driven Deep Reinforcement Learning’ using the maze environments.

### Installation
  ```Shell
  sudo apt-get install -y tmux htop cmake golang libjpeg-dev
  git clone -b master https://github.com/m-fritsche/value-prediction-consistency.git
  conda create -n curiosity python=2.7
  source activate curiosity
  pip install numpy
  pip install -r value-prediction-consistency/requirements.txt
  ```

### Usage
  ```Shell
  cd value-prediction-consistency/
  # for A3C remove --unsup, for A3C + Pred use --unsup pred, for A3C + Pred + VPC use --unsup vpc
  # for Maze A use --env-id mazeSmall-0, for Maze B use --env-id mazeLarge-v0
  python train.py --unsup vpc --env-id mazeSmall-v0
  ```
  Training process is shown in Tensorboard on http://localhost:12345

### Acknowledgement
The implentation is based on the code of [Curiosity-driven Exploration by Self-supervised Prediction](https://github.com/pathak22/noreward-rl).  
Vanilla A3C code is based on the open source implementation of [universe-starter-agent](https://github.com/openai/universe-starter-agent).  
Maze implementations are based on [Pycolab](https://github.com/deepmind/pycolab)
