from utils.visualization import show_hps_behaviour
import matplotlib.pyplot as plt

fig_path = "images/comparison_ablation_MixMSE.pdf"
n_rows = 2
n_cols = 4

fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*5, n_rows*5))

show_hps_behaviour(root="results/day-17-03-2023_hr-12-02-54_epochs-12_batchsize-500_ABLATION_AB",
                   axs=axs[0])

show_hps_behaviour(root="results/day-24-03-2023_hr-19-41-25_AT_FGSM_ABLATION",
                   axs=axs[1])

axs[0][0].set_ylabel("MixMSE")
axs[1][0].set_ylabel("MixMSE-FGSM-AT")

fig.show()
fig.savefig(fig_path)
print("")
# show_hps_behaviour(root="results/day-17-03-2023_hr-12-02-54_epochs-12_batchsize-500_ABLATION_AB",
#                    fig_path="images/MixMSE-ablation.pdf")

# show_hps_behaviour(root="results/day-24-03-2023_hr-19-41-25_AT_FGSM_ABLATION",
#                    fig_path="images/MixMSE-AT-ablation.pdf")
