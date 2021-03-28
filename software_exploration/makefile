all: envs

envs: requirements.yml
	conda env create --prefix ./envs -f requirements.yml

test:
	sbatch run_script.sh

