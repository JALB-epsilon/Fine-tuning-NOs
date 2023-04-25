from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
import yaml
import argparse 
import utilities
import os
import torch
import shutil

def datasetFactory(config, do, args=None):
    c_data =config["data"]
    if args is None:
        gl = utilities.GettingLists(data_for_training=c_data["n_sample"],
                                    wave_eq = c_data["PDE_type"],
                                    data_base = c_data["process"], 
                                    PATH = c_data["PATH"])
        return utilities.MyLoader(GL=gl, do = do, config=config, args = None)
    elif args is not None:
        gl = utilities.GettingLists(data_for_training=c_data["n_sample"],
                                    wave_eq = c_data["PDE_type"],
                                    data_base = args.data_base, 
                                    PATH = args.PATH)
        return utilities.MyLoader(GL=gl, do = do, config=config, args=args)


def main(args, config = None):
    if config is None:
        with open(args.config_file, 'r') as stream:
            config = yaml.load(stream, yaml.FullLoader)
    print(config) 
    print(args)
    model = utilities.choosing_model(config)
    print(model)

    if args.all_ckp == False:
        save_file = os.path.join(config["ckpt"]["PATH"], 
                                config["ckpt"]["save_dir"]
                                )
        checkpoint_callback = ModelCheckpoint(                                
                                dirpath=save_file,
                                every_n_epochs = 1,
                                save_last = True,
                                monitor = 'val_loss',
                                mode = 'min',
                                save_top_k = args.save_top_k,
                                filename="model-{epoch:03d}-{val_loss:.4f}",
                            )

    elif  args.all_ckp == True:
        save_file = os.path.join(config["ckpt"]["PATH"], "all_epochs",
                                        f'{config["train"]["epochs"]}_{config["model"]["activ"]}',
                                        config["ckpt"]["save_dir"]
                                )
        checkpoint_callback = ModelCheckpoint(                                
                                dirpath=save_file,
                                every_n_epochs = 1,
                                save_top_k = -1,
                                filename="model-{epoch:03d}-{val_loss:.4f}",
                            )

    if os.path.exists(save_file):
        print(f"The model directory exists. Overwrite? {args.erase}")
        if args.erase == True:
            shutil.rmtree(save_file)

    if args.checkpoint is None:
        #left the default values provided by the config file
        train_dataloader, val_dataloader = datasetFactory(config=config, do=args.do, args=None)
        max_epochs = config["train"]["epochs"]
    elif args.checkpoint is not None:
        print(f"Load from checkpoint {args.checkpoint}")  
        model=model.load_from_checkpoint(args.checkpoint)
        #change optimizer if needed
        if args.optimizer is not None:
            if args.optimizer == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
            elif args.optimizer == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        else:
            if args.weight_decay is None:
                args.weight_decay = config["train"]["weight_decay"]
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        #change the scheduler if needed
        if args.scheduler is not None:
            if args.scheduler == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True, eps=1e-08, min_lr=0)
            elif args.scheduler == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0, last_epoch=-1)
        else:
            if args.step_size is None:
                args.step_size = config["train"]["step_size"]
            if args.gamma is None:
                args.gamma = config["train"]["gamma"]
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
        model.configure_optimizers(optimizer, scheduler)
        #change the number of epochs if needed
        if args.epochs is not None:
            print(f"Change the number of epochs to {args.epochs}")
            max_epochs = args.epochs
            
        #change the filename if needed
        if args.filename is not None:
            print(f"Change the filename to {args.filename}")
            checkpoint_callback.filename = args.filename+"-{epoch:03d}-{val_loss:.4f}"

        if args.data_base is None:
            args.data_base = config["data"]["process"]
            args.PATH = config["data"]["PATH"]
        else: 
            args.PATH = os.path.join('save_files', 'acoustic', args.data_base)
        #change the config file if needed through the command line
        train_dataloader, val_dataloader = datasetFactory(config=config, do = args.do, args=args)
        
    if args.usual_ckpt == True:
        trainer = pl.Trainer(max_epochs=max_epochs,
                    accelerator=args.accelerator, 
                    devices=args.devices,
                    default_root_dir=save_file) 

    elif args.usual_ckpt == False:
        trainer = pl.Trainer(max_epochs=max_epochs,
                            accelerator=args.accelerator, 
                            devices=args.devices,
                            callbacks=[checkpoint_callback]) 

    trainer.fit(model, train_dataloader, val_dataloader)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Training of the Architectures', add_help=True)
    parser.add_argument('-c','--config_file', type=str, 
                                help='Path to the configuration file',
                                default='config/acoustic/GRF_7Hz/FNO25k.yaml')
    parser.add_argument('-a', '--all_ckp', type = bool, 
                                 help='Allow to save all the ckpt',
                                default=False)
    parser.add_argument('-u_ckpt', '--usual_ckpt', type = bool,
                                help='Allow to save the usual ckpt as in pytorch-lightning\
                                use it only if you want to save the ckpt in a different directory\
                                    or multiple analysis of networks',
                                default=False)
    parser.add_argument('-e', '--erase', type = bool, 
                                 help='erase_save_dir',
                                default=False)
    parser.add_argument( '-ckpt', '--checkpoint', type = str, 
                                help='checkpoint file to load',    
                                default=None)
    parser.add_argument('-savetop', '--save_top_k', type = int, 
                                help='save top k ckpt',
                                default=3)
    parser.add_argument('-lr', '--lr', type = float,
                                help='learning rate',       
                                default=None)
    parser.add_argument('-o', '--optimizer', type = str,
                                help='optimizer',   
                                default=None)
    parser.add_argument('-s', '--scheduler', type = str,
                                help='scheduler',       
                                default=None)
    parser.add_argument('-ep', '--epochs', type = int,
                                help='number of epochs',            
                                default=None)
    parser.add_argument('-f', '--filename', type = str,
                                help='filename',
                                default=None)
    parser.add_argument('-b', '--batch_size', type = int,
                                help='batch_size',  
                                default=None)
    parser.add_argument('-lw', '--load_workers', type = int,
                                help='load_workers',    
                                default=None)
    parser.add_argument('-db', '--data_base', type = str,
                                help='database',
                                default=None)
    parser.add_argument('-weight_decay', '--weight_decay', type = float,
                                help='weight_decay',
                                default=None)
    parser.add_argument('-step_size', '--step_size', type = int,
                                help='step_size',   
                                default=None)
    parser.add_argument('-gamma', '--gamma', type = float,
                                help='gamma',
                                default=None)
    parser.add_argument('-P', '--PATH', type = str,
                                help='PATH',        
                                default=None)
    parser.add_argument('-d', '--devices', type = int,
                                help='devices', 
                                default=1)
    parser.add_argument('-acc', '--accelerator', type = str,
                                help='accelerator',
                                default='gpu')
    parser.add_argument('-do', '--do', type=str,
                            help='do',  
                            default="train")
    

    args=parser.parse_args()
    config_file = args.config_file
    main(args)