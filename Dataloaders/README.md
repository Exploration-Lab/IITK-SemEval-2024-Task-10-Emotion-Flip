### Instructions
- For the Task of ERC, only run the `ERC_Dataloader.py` to create the pickles
- For the Task of EFR, run the `ERC_Dataloader.py`, followed by the `EFR_Dataloader.py`
 
### ERC_Dataloader.py
- Create a Folder named Pickles with `sent2emb.pickle` present in it
- Update the variables `training_path`, `testing_path` and `save_path` to the training csv, testing csv and the pickles folder
- Update the variables `emos`, `seq_len` and `emb_size` according to the emotions in the data and the model parameters

### EFR_Dataloader.py
- Update the variables `training_path`, `testing_path` and `save_path` to the training csv, testing csv and the pickles folder
- Update the variables `seq_len` according to the model parameters
