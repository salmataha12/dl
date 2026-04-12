# ConvMLP/early_stopping.py
"""
Early Stopping for ConvMLP models
"""

class ConvMLPEarlyStopping:
    """Early stopping to avoid overfitting"""
    
    def __init__(self, patience=10, verbose=False, delta=0.0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_acc, epoch):
        """
        Args:
            val_acc: Validation accuracy
            epoch: Current epoch
            
        Returns:
            True if training should stop, False otherwise
        """
        score = val_acc

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.counter = 0
            if self.verbose:
                print(f'✓ Val Acc improved to {score:.4f}')
        else:
            self.counter += 1
            if self.verbose:
                print(f'No improvement. Counter {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False