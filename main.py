from utils.train import train_model
from utils.visualizer import plot_training_curve, plot_weight_distribution

def main():
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(batch_size=128)
    plot_training_curve(train_losses, val_losses, train_accuracies, val_accuracies)
    plot_weight_distribution()
    


if __name__ == "__main__":
    main()




