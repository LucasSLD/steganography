*This file contains informations about what needs to be done for the transfer learning of SRNet using the code provided by Jan Butora.*

## <u>**LitModel.py**</u><br>
This file contains the full model used for transfer learning on SRNet. It follows LightningModule way of organizing the code.

### Learning rate
The way the learning rate evolves during training can be set with *lr_scheduler_name* parameter (line 201 configure_optimizers function).

### IL_train & IL_test
I must update the **WORK_DIR** constant used in this file (it is used line 266 and 268).<br>
The files IL_val.p, IL_train.p and IL_test.p should be in $WORK/DataBase/BOSSBase512/<br>
Either I do all changes in code by changing hard coded path with the path of my directories and I change **WORK_DIR** or I just change **WORK_DIR** and modify my folder structure. <span style="color: red">**I will probably change the code and not the directories.**</span>
