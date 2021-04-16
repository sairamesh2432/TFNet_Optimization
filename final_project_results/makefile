all: envs

envs: requirements.yml
	conda env create --prefix ./envs -f requirements.yml

lightning_test:
	sbatch run_lightning.sh
    
torch_test:
	sbatch run_base_torch.sh

