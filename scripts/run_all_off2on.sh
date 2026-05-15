eval "$(conda shell.bash hook)"
conda activate ail

name="run_off2on"
seed=2

CUDA_VISIBLE_DEVICES=0 python off2on.py env=finger_spin agent=foil expert.demos=50 method=il seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=0 python off2on.py env=cheetah_run agent=foil expert.demos=50 method=il seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=0 python off2on.py env=hopper_hop agent=foil expert.demos=10 method=il seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=0 python off2on.py env=hopper_stand agent=foil expert.demos=10 method=il seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=1 python off2on.py env=cartpole_swingup agent=foil expert.demos=10 method=il seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=1 python off2on.py env=walker_walk agent=foil expert.demos=40 method=il seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=1 python off2on.py env=walker_stand agent=foil expert.demos=30 method=il seed="$seed" agent.bc_transit=false project.name="$name" &
sleep 2
CUDA_VISIBLE_DEVICES=1 python off2on.py env=walker_run agent=foil expert.demos=50 method=il seed="$seed" agent.bc_transit=false project.name="$name" &
wait
