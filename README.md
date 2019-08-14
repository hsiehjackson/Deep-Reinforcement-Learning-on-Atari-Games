# [107-2] Applied Deep Learning - Deep Reinforcement Learning on Atari Games
In this project, we learn how to implement several agents to play [Atari Games](https://gym.openai.com/envs/#atari) including Policy Gradient, Deep Q-Learning (DQN), and Advantange-Actor-Critic (a2c).

## Usage
* Git clone the code and install package
```
git clone https://github.com/hsiehjackson/
pip install -r requirements
```
* Download files and extract zip file
```
bash download.sh
unzip download.zip
```
### Policy Gradient
* Training
```
python main.py --train_pg --pg_type=pg || pg_nor || pg_ppo --folder_name=[YOUR NAME]
```
* Testing (your/my best)
```
python main.py --test_pg --model_path=[YOUR MODEL PATH]
python main.py --test_pg --model_path=./model/best/pg.cpt
```
### Deep Q-Learning
* Training
```
python main.py --train_dqn --dqn_type=DQN || DoubleDQN || DuelDQN || DDDQN --folder_name=[YOUR NAME]
```
* Testing (your/my best)
```
python main.py --test_dqn --dqn_type=DQN || DoubleDQN || DuelDQN || DDDQN --model_path=[YOUR MODEL PATH]
python main.py --test_dqn --dqn_type=DDDQN --model_path=./model/best/dqn.cpt
```

### Advantange-Actor-Critic
* Training
```
python main.py --train_a2c --folder_name=[YOUR NAME]
```
* Testing (your/my best)
```
python main.py --test_a2c --model_path=[YOUR MODEL PATH]
python main.py --test_a2c --model_path=./model/best/a2c.cpt
```
### Others
* Plot training progress
```
python plot.py ./model/[folder_name]/plot.json
```

## Environment Introdution
We use three Atari Games to test our performance separately, such as [LunarLander](https://en.wikipedia.org/wiki/Lunar_Lander_(1979_video_game)), [Assault](https://en.wikipedia.org/wiki/Assault_(1983_video_game)), and [Mario](https://en.wikipedia.org/wiki/Super_Mario). You can click the hyperlinks to see the game rules. The following GIFs are my best results.

| LunarLander  | Assault | Mario |
| :--------: | :--------: | :--------: |
|![](image/Lunarlander.gif | height=200) |![](image/Assault.gif | height=200) |![](image/Mario.gif | height=200)| 

## Techniques for Deep Reinforcement Learning
### Policy Gradient
We implement policy gradient agents with REIFORCE algorithm. However, I also use some improvements including reward normalization and proximal policy optimization (PPO).
* Reward Normalization

Due to all positive rewards, we can subtract a baseline (normalized) to let **rewards have negative value**. With baseline, the probability of the not sampled actions will not decrease sharply

* Proximal policy optimization
PPO had implemented off-policy algorithm with **important sampling**, which set KL divergence constraints for that θ cannot very different from θ'. The objective function is shown below.

<img src="https://i.imgur.com/EnWmRf7.png"/>

### Deep Q-Learning (DQN)
Besides classic DQN algorithm, we also implement some simple improvements for DQN, such as Double DQN and Dueling DQN.
* Double DQN

It is used to **solve over-estimated problems**. With two networks (online and target), they can compensate for the other to avoid over-estimated q value. This method is only need another copied network to acquire target actions.
* Dueling DQN

Dueling DQN is used to acquire the state q value among each actions and **set a normalized constraint** for different action q value. With this method, we can update the action even if we don’t sample on it, which is more efficient.

| Double DQN | Dueling DQN|
| :--------: | :--------: |
| <img src="https://i.imgur.com/bkSeAOk.png"  height="120" /> | <img src="https://i.imgur.com/l0PGdm8.png"  height="120" />|

### Advantage-Actor-Critic (A2C) 

Different from general a2c framework, we also implement proximal policy optimization (PPO) and **Generalized Advantage Estimation (GAE)** on multi-processing environment with value loss, action loss, and entropy loss. This method can consider the KL-divergence constraints and train more iteration on one steps. Our framework was reference from [here](https://github.com/higgsfield/RL-Adventure-2/blob/master/3.ppo.ipynb).


## Training Procedure Results
It is obvious that **reward normalization** can have better results than baselin. However, PPO cannot show little improvements but only reduce the unstable variance. Perhaps its ability isn't apparent on LunarLander. 

| Policy Gradient |
|:----------:|
|<img src="https://i.imgur.com/SkeWHhq.png" />|

While we can find better results than baseline DQN with several improved techniques, the joint use of DoubleDQN and DuelingDQN cannot get the highest performance. These results suggest **DuelingDQN has a great ability** to improve DQN framework on Assault.

| Deep Q-Learning (DQN) |
|:----------:|
|<img src="https://i.imgur.com/FVbXgZ0.png" />|

As for the results of A2C, we can find that the performance **with PPO is higher than without PPO**, which is different from policy gradient results. Without PPO, the performance will go decayed after more training steps, showing the difficulty to learn a good agent on Mario.

| Advantage-Actor-Critic (A2C) |
|:----------:|
|<img src="https://i.imgur.com/jeTrEza.png" />|


## Reward Results

|| Policy Gradient | Deep Q-Learning | Advantange-Actor-Critic |
| :--------: | :--------: | :--------: | :--------: |
|Games| LunarLander | Assault | Mario |
|Test Episodes|30|100|10|
|Reward|90.11|275.95|3243.30|
