eval "$(conda shell.bash hook)"
conda activate ail

name="run"
seed=2

CUDA_VISIBLE_DEVICES=0 python train.py env=finger_spin agent=ail expert.demos=50 method=il method.lambda_gp=10 seed="$seed" project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=0 python train.py env=cheetah_run agent=ail expert.demos=50 method=il method.lambda_gp=10 seed="$seed" project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=0 python train.py env=hopper_hop agent=ail expert.demos=10 method=il method.lambda_gp=10 seed="$seed" project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=0 python train.py env=hopper_stand agent=ail expert.demos=10 method=il method.lambda_gp=10 seed="$seed" project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=1 python train.py env=cartpole_swingup agent=ail expert.demos=10 method=il method.lambda_gp=1 seed="$seed" project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=1 python train.py env=walker_walk agent=ail expert.demos=10 method=il method.lambda_gp=1 seed="$seed" project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=1 python train.py env=walker_stand agent=ail expert.demos=10 method=il method.lambda_gp=1 seed="$seed" project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=1 python train.py env=walker_run agent=ail expert.demos=10 method=il method.lambda_gp=10 seed="$seed" project.name="$name" &
wait
