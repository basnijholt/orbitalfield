env_path = "/home/main/anaconda2/envs/python3/bin/python"
c.LocalControllerLauncher.controller_cmd = [env_path, "-m", "ipyparallel.controller"]
c.LocalEngineLauncher.engine_cmd = [env_path, "-m", "ipyparallel.engine"]
c.LocalEngineSetLauncher.engine_cmd = [env_path, "-m", "ipyparallel.engine"]
