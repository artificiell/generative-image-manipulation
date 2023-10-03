import torch
import matplotlib.pyplot as plt

plt.style.use('ggplot')

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(self, best_valid_loss=float('inf')):
        self.best_valid_loss = best_valid_loss
        
    def __call__(self, current_valid_loss, epoch, model, optimizer, criterion, model_name):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch}\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, './outputs/{}_best_model.pth'.format(model_name))
            

def save_model(epoch, model, optimizer, criterion, model_name):
    """
    Function to save the trained models to disk.
    """
    print(f"\nSaving model at epoch: : {epoch}\n")
    torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, './outputs/{}_epoch_{}.pth'.format(model_name, epoch))


def save_accuracy_plot(train_acc, valid_acc):
    """
    Function to save the accuracy plots to disk.
    """
    # accuracy plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_acc, color='green', linestyle='-', 
        label='train accuracy'
    )
    plt.plot(
        valid_acc, color='blue', linestyle='-', 
        label='validataion accuracy'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('./outputs/accuracy.png')

def save_loss_plot(train_loss, valid_loss, test_loss):
    """
    Function to save the loss plots to disk.
    """    
    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(
        train_loss.keys(), train_loss.values(),
        color='blue', linestyle='-', label='Train loss'
    )
    plt.plot(
        valid_loss.keys(), valid_loss.values(),
        color='red', linestyle='-', label='Validataion loss'
    )
    plt.plot(
        test_loss.keys(), test_loss.values(),
        color='orange', linestyle='-', label='Test loss'
    )
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./outputs/loss.png')
