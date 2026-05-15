eval "$(conda shell.bash hook)"
conda activate ail

name="run_pre"
seed=2

CUDA_VISIBLE_DEVICES=0 python off2on.py env=finger_spin agent=ail expert.demos=50 method=il method.lambda_gp=10 seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=0 python off2on.py env=cheetah_run agent=ail expert.demos=50 method=il method.lambda_gp=10 seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=0 python off2on.py env=hopper_hop agent=ail expert.demos=10 method=il method.lambda_gp=10 seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=0 python off2on.py env=hopper_stand agent=ail expert.demos=10 method=il method.lambda_gp=10 seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=1 python off2on.py env=cartpole_swingup agent=ail expert.demos=10 method=il method.lambda_gp=1 seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=1 python off2on.py env=walker_walk agent=ail expert.demos=10 method=il method.lambda_gp=1 seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=1 python off2on.py env=walker_stand agent=ail expert.demos=10 method=il method.lambda_gp=1 seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=1 python off2on.py env=walker_run agent=ail expert.demos=10 method=il method.lambda_gp=10 seed="$seed" agent.bc_transit=false project.name="$name" &
wait
