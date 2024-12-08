import pandas as pd
import re

def process_eval_loss(txt_file_path, output_csv_file):
    data = []
    eval_loss_pattern = re.compile(
        r"type: eval_loss, epoch: (\d+), step: (\d+).*?elbo: ([\d.]+).*?elbo_filtered: ([\d.]+)"
    )

    # Read and process the text file
    with open(txt_file_path, 'r') as file:
        for line in file:
            match = eval_loss_pattern.search(line)
            if match:
                # Extract relevant information
                epoch, step, elbo, elbo_filtered = match.groups()
                data.append({
                    "epoch": int(epoch),
                    "step": int(step),
                    "elbo": float(elbo),
                    "elbo_filtered": float(elbo_filtered)
                })

    df = pd.DataFrame(data)
    df.to_csv(output_csv_file, index=False)
    print(f"CSV file saved to {output_csv_file}")

process_eval_loss('training_log_cifar.txt', 'fixed_cifar_loss_log.csv')