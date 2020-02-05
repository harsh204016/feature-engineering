import sandesh
from  keras.callbacks import Callback

class LossHistory(Callback):
    
    def on_train_begin(self, logs={}):
        self.losses = []

    
    def on_epoch_end(self, epoch, logs={}):
        
        loss=logs.get('loss')
        acc=logs.get('acc')
        msg = f"Training accuracy was {acc} Training loss was {loss}"
        sandesh.send(msg,webhook="https://hooks.slack.com/services/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")
        
