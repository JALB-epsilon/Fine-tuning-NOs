from models import *

def choosing_model(config):
    c_nn = config["model"]
    c_train = config["train"]
    # 7 Hz data only contains the real part of the field
    if config["Project"]["database"]=='GRF_7Hz':
        if config["Project"]["name"] == "FNO":         
            model =FNO(
                    wavenumber = c_nn["modes_list"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"],
                    activation = c_nn["activ"]
                    )

        elif config["Project"]["name"] == "sFNO":
            model =sFNO(
                    wavenumber = c_nn["modes_list"],
                    drop = c_nn["drop"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"],
                    activation = c_nn["activ"]
                        )
        elif config["Project"]["name"] == "FNO_residual":
            model =FNO_residual(
                    wavenumber = c_nn["modes_list"],
                    drop = c_nn["drop"],
                    drop_path = c_nn["drop_path"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"],
                    activation = c_nn["activ"]
                    )

        
        elif config["Project"]["name"] == "sFNO+epsilon_v1":
            model =sFNO_epsilon_v1(
                    wavenumber = c_nn["modes_list"],
                    drop = c_nn["drop"],
                    drop_path = c_nn["drop_path"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"],
                    activation = c_nn["activ"]
                    )


        elif config["Project"]["name"] == "sFNO+epsilon_v2":
            model =sFNO_epsilon_v2( 
                modes = c_nn["modes_list"],
                drop_path_rate = c_nn["drop_path"],
                drop = c_nn["drop"],
                depths = c_nn["depths"], 
                dims = c_nn["dims"],
                learning_rate = c_train["lr"], 
                step_size= c_train["step_size"],
                gamma= c_train["gamma"],
                weight_decay= c_train["weight_decay"],
                activation = c_nn["activ"]
                )
    # 12/15 Hz data only contains real and imaginary part of the field
    elif config["Project"]["database"]==('GRF_12Hz') or ('GRF_15Hz'):               
        
        if config["Project"]["name"] == "FNO":         
            Proj = torch.nn.Linear(c_nn["features"], 2, dtype=torch.float)
            model =FNO(
                    wavenumber = c_nn["modes_list"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"],
                    proj = Proj,
                    activation = c_nn["activ"]
                    )

        elif config["Project"]["name"] == "sFNO":
            Proj = torch.nn.Linear(c_nn["features"], 2, dtype=torch.float)
            model =sFNO(
                    wavenumber = c_nn["modes_list"],
                    drop = c_nn["drop"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"],
                    proj = Proj,
                    activation = c_nn["activ"]
                    )


        elif config["Project"]["name"] == "FNO_residual":
            Proj = torch.nn.Linear(c_nn["features"], 2, dtype=torch.float)
            model =FNO_residual(
                    wavenumber = c_nn["modes_list"],
                    drop = c_nn["drop"],
                    drop_path = c_nn["drop_path"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"],
                    proj = Proj,
                    activation = c_nn["activ"]
                        )

        elif config["Project"]["name"] == "sFNO+epsilon_v1":
            Proj = torch.nn.Linear(c_nn["features"], 2, dtype=torch.float)
            model =sFNO_epsilon_v1(
                    wavenumber = c_nn["modes_list"],
                    drop = c_nn["drop"],
                    drop_path = c_nn["drop_path"],
                    features_ = c_nn["features"],
                    learning_rate = c_train["lr"], 
                    step_size= c_train["step_size"],
                    gamma= c_train["gamma"],
                    weight_decay= c_train["weight_decay"],
                    proj = Proj,
                    activation = c_nn["activ"]
                        )

        elif config["Project"]["name"] == "sFNO+epsilon_v2":
            #sFNO_epsilon_v2_proj is the same arch as sFNO_epsilon_v2
            Proj = torch.nn.Linear(c_nn["dims"][-1], 2, dtype=torch.float)
            model =sFNO_epsilon_v2_proj( 
                modes = c_nn["modes_list"],
                drop_path_rate = c_nn["drop_path"],
                drop = c_nn["drop"],
                depths = c_nn["depths"], 
                dims = c_nn["dims"],
                learning_rate = c_train["lr"], 
                step_size= c_train["step_size"],
                gamma= c_train["gamma"],
                weight_decay= c_train["weight_decay"],
                proj = Proj,
                activation = c_nn["activ"]
                )
        elif config["Project"]["name"] == "sFNO+epsilon_v2_updated":
            #sFNO_epsilon_v2_proj is the same arch as sFNO_epsilon_v2
            #we just allow to have an independent projection layer.
            #Proj = torch.nn.Linear(c_nn["dims"][-1], 2, dtype=torch.float)
            model =sFNO_epsilon_v2_updated( 
                stage_list = c_nn["depths"], 
                features_stage_list = c_nn["dims"],
                wavenumber_stage_list = c_nn["modes_list"],
                dim_input = 1,
                dim_output = 2,
                
                #proj = Proj,
                activation = c_nn["activ"],
                drop_rate = c_nn["drop"], 
                drop_path_rate = c_nn["drop_path"],               
                learning_rate = c_train["lr"], 
                step_size= c_train["step_size"],
                gamma= c_train["gamma"],
                weight_decay= c_train["weight_decay"],
                )

    return model