import pandas as pd

def plot_loss(loss, ax, window=20):
    loss_df = pd.DataFrame(loss)

    if len(loss['tot']) < window*10:
        window = 1
    
    if isinstance(window, int):
        loss_df = loss_df.rolling(window).mean()

    loss_df.plot(ax=ax)
    ax.legend(fontsize=15)
    ax.set_xlabel('iterations')