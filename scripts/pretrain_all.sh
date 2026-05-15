eval "$(conda shell.bash hook)"
conda activate ail

seed=2

CUDA_VISIBLE_DEVICES=0 python pretrain.py env=finger_spin agent=bc expert.demos=50 env.learn_steps=1e5 seed="$seed"

CUDA_VISIBLE_DEVICES=0 python pretrain.py env=cheetah_run agent=bc expert.demos=50 env.learn_steps=1e5 seed="$seed"

CUDA_VISIBLE_DEVICES=0 python pretrain.py env=hopper_hop agent=bc expert.demos=10 env.learn_steps=1e5 seed="$seed"

CUDA_VISIBLE_DEVICES=0 python pretrain.py env=hopper_stand agent=bc expert.demos=10 env.learn_steps=1e5 seed="$seed"

CUDA_VISIBLE_DEVICES=0 python pretrain.py env=cartpole_swingup agent=bc expert.demos=10 env.learn_steps=1e5 seed="$seed"

CUDA_VISIBLE_DEVICES=0 python pretrain.py env=walker_walk agent=bc expert.demos=10 env.learn_steps=1e5 seed="$seed"

CUDA_VISIBLE_DEVICES=0 python pretrain.py env=walker_stand agent=bc expert.demos=10 env.learn_steps=1e5 seed="$seed"

CUDA_VISIBLE_DEVICES=0 python pretrain.py env=walker_run agent=bc expert.demos=10 env.learn_steps=1e5 seed="$seed"
