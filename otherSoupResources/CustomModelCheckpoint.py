import torch
from pytorch_lightning.callbacks import ModelCheckpoint


class CustomModelCheckpoint(ModelCheckpoint):
    def save_checkpoint(self, trainer, pl_module):
        # Get the current epoch number
        epoch = trainer.current_epoch
        # Call the original save_checkpoint method
        super().save_checkpoint(trainer, pl_module)
        # Modify the saved checkpoint file to include the epoch number and optimizer/scheduler states
        filepath = self.format_checkpoint_name(epoch, self.current_score)
        checkpoint = torch.load(filepath)
        checkpoint['epoch'] = epoch
        # Save optimizer and scheduler states
        checkpoint['optimizer_state_dict'] = trainer.optimizer.state_dict()
        checkpoint['lr_scheduler_state_dict'] = trainer.lr_schedulers[0]['scheduler'].state_dict()

        torch.save(checkpoint, filepath)
def custom_load_checkpoint(model,checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    epoch = checkpoint['epoch']
    my_model = model.load_from_checkpoint(checkpoint_path)
    return my_model, epoch
    #optimizer_state_dict = checkpoint['optimizer_state_dict']
    #scheduler_state_dict = checkpoint['scheduler_state_dict']
    # Load the components as needed
    # ...


    #trainer.optimizer.load_state_dict(optimizer_state_dict)
    #trainer.lr_schedulers[0]['scheduler'].load_state_dict(scheduler_state_dict)
