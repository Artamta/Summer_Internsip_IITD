import pandas as pd

# Load your results
df = pd.read_csv("/Users/ayush/Desktop/project-internsip/reference_maps/accuracy_metrics_all_simulations.csv")

# Define parameters and their simulation numbers
param_sim_map = {
    'D': 1,
    'D_star': 2,
    'f': 3,
    'k': 4
}
snr_order = [60, 40, 25, 15]
data_order = [1, 2, 3, 4, 5]

# Create a writer for Excel
with pd.ExcelWriter("/Users/ayush/Desktop/project-internsip/reference_maps/simulation_results_supervisor_format.xlsx") as writer:
    for param, sim_num in param_sim_map.items():
        rows = []
        for snr in snr_order:
            for data in data_order:
                row = df[
                    (df['Simulation'] == sim_num) &
                    (df['Parameter'] == param) &
                    (df['SNR'] == snr) &
                    (df['Data'] == data)
                ]
                if not row.empty:
                    row = row.iloc[0]
                    rows.append([
                        f"SNR{snr}" if data == 1 else "",
                        data,
                        row['RMSE_norm'],
                        row['Rel_Bias'],
                        row['Rel_Param'],
                        row['AIC'],
                        row['AICc']
                    ])
                else:
                    rows.append([
                        f"SNR{snr}" if data == 1 else "",
                        data, "", "", "", "", ""
                    ])
        # Create DataFrame for export
        export_df = pd.DataFrame(rows, columns=[
            "SNR", "Data", "RMSE(%)", "Relative Bias (%)", "Relative Parameter", "Akaike Information criteria (AIC)", "AIC Corrected"
        ])
        # Write each parameter to a separate sheet
        export_df.to_excel(writer, sheet_name=param[:31], index=False)

print("Saved Excel file in supervisor's format with separate sheets for D, D_star, f, k.")