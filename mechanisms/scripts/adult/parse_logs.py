import re

name_to_log = {
    'AIM-L1': 'aim_l1_eps_1.0.log',
    'AIM-L2': 'aim_l2_eps_1.0.log',
    'FHAIM-L1': 'fhaim_l1_eps_1.0.log',
    'FHAIM-L2': 'fhaim_l2_eps_1.0.log',
}

base_log_path = "../../../data/logs/run_id/adult/"

for i in range(0, 6):
    print(f"For Run ID: {i}")
    for k, v in name_to_log.items():
        log_path = base_log_path.replace("run_id", f"run_{i}") + v

        try:
            with open(log_path, 'r') as f:
                content = f.read()
        except FileNotFoundError:
            print(f"Log file not found: {log_path}")
            continue

        error_match = re.search(r'Average Error:\s+([\d.eE+-]+)', content)
        time_match = re.search(r'Time taken to train and generate:\s+([\d.eE+-]+)\s+minutes', content)

        error = error_match.group(1) if error_match else "N/A"
        runtime = time_match.group(1) if time_match else "N/A"

        print(f"For Technique : {k}, Error = {error}, Runtime: {runtime} minutes")

    print("=====================================================")
