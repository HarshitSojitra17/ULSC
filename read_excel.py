import pandas as pd
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

base = 'DJSCC'

groups = {
    '7sm':  [f'7sm_results_{p}.xlsx' for p in ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']],
    '7PSNR': [f'7PSNR_results_{p}.xlsx' for p in ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']],
    '14sm': [f'14sm_results_{p}.xlsx' for p in ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']],
    '14psnr': [f'14psnr_results_{p}.xlsx' for p in ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']],
    'NO7sm': [f'NO7sm_results_{p}.xlsx' for p in ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']],
    'NO14sm': [f'NO14sm_results_{p}.xlsx' for p in ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']],
    'NO7PSNR': [f'NO7PSNR_results_{p}.xlsx' for p in ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']],
    'NO14psnr': [f'NO14psnr_results_{p}.xlsx' for p in ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']],
    'Attacker_7PSNR': [f'Attacker_7PSNR_results_{p}.xlsx' for p in ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']],
    'Attacker_NO7PSNR': [f'Attacker_NO7PSNR_results_{p}.xlsx' for p in ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']],
    'Attacker_14PSNR': [f'Attacker_14PSNR_results_{p}.xlsx' for p in ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']],
    'Attacker_NO14PSNR': [f'Attacker_NO14PSNR_results_{p}.xlsx' for p in ['0.001','0.004','0.007','0.01','0.04','0.07','0.1']],
}

for group_name, files in groups.items():
    print(f"\n===== {group_name} =====")
    first_file = os.path.join(base, files[0])
    if os.path.exists(first_file):
        df = pd.read_excel(first_file)
        cols = [str(c) for c in df.columns.tolist()]
        print(f"  Columns: {cols}")
        print(f"  Shape: {df.shape}")
        # Print mean of each numeric column
        for col in df.select_dtypes(include='number').columns:
            mean_val = df[col].mean()
            print(f"  [{col}] mean = {mean_val:.4f}")
    else:
        print(f"  NOT FOUND: {first_file}")
