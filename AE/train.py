import torch
import torch.nn as nn
import numpy as np




def train(
    model,
    epochs,
    train_loader,
    val_loader,
    optimizer,
    writer,
    scheduler=None,
    save_tensorboard_parameters=False,
    starting_epoch = 0
):
    global_batch_idx = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(model.device)
            optimizer.zero_grad()
            output = model(data)
            loss = nn.MSELoss()(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            global_batch_idx += 1

        if scheduler is not None:
            scheduler.step()

        writer.add_scalar(
            "Loss/train", train_loss / len(train_loader.dataset), global_step=(epoch + starting_epoch)
        )

        print(
            "Epoch: {}/{}, Average loss: {:.4f}".format(
                epoch, epochs, train_loss / len(train_loader.dataset)
            )
        )

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(val_loader):
                data = data.to(model.device)
                output = model(data)
                loss = nn.MSELoss()(output, data)
                val_loss += loss.item()

        writer.add_scalar(
            "Loss/val", val_loss / len(val_loader.dataset), global_step=(epoch + starting_epoch)
        )

        if save_tensorboard_parameters:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, global_step=epoch)
                writer.add_histogram(f"{name}.grad", param.grad, global_step=epoch)

    writer.close()
    print(
        f"Training completed. Final training loss: {train_loss / len(train_loader.dataset)}, Validation loss: {val_loss / len(val_loader.dataset)}"
    )





# ------------------------------- Layer wise pretrain funcs ------------------------------

def layer_wise_pretrain_load_dict(ex_model, new_model):
    for i in range(len(ex_model.encoder)):
        if isinstance(ex_model.encoder[i], nn.Linear):
            # Get the corresponding layer in the new model
            for j in range(len(new_model.encoder)):
                if isinstance(new_model.encoder[j], nn.Linear):
                    if ex_model.encoder[i].weight.shape == new_model.encoder[j].weight.shape:
                        new_model.encoder[j].weight.data = ex_model.encoder[i].weight.data.clone()
                        new_model.encoder[j].bias.data = ex_model.encoder[i].bias.data.clone()
                        break

    # For the decoder (in reverse order since decoder is reversed)
    for i in range(len(ex_model.decoder)):
        if isinstance(ex_model.decoder[i], nn.Linear):
            # Get the corresponding layer in the new model
            for j in range(len(new_model.decoder)):
                if isinstance(new_model.decoder[j], nn.Linear):
                    if ex_model.decoder[i].weight.shape == new_model.decoder[j].weight.shape:
                        new_model.decoder[j].weight.data = ex_model.decoder[i].weight.data.clone()
                        new_model.decoder[j].bias.data = ex_model.decoder[i].bias.data.clone()
                        break



# -------------------------------- Training one feature at the time ----------------------------
 # neurons are 1-indexed


# ------------- Functions to mask parameters ------------------------

def mask_bottleneck_in_weight(weight, neuron_idx):      
    with torch.no_grad():
        active_neurons = list(range(neuron_idx))
        train_neuron = active_neurons[-1]
        mask_mul = torch.zeros_like(weight)
        mask_add = torch.zeros_like(weight)
        mask_mul[active_neurons, :] = 1                 # weights have shape (latent_dim, encoder[-1])
        mask_add[train_neuron, :] = torch.randn((1, weight.size(1)))
        return weight * mask_mul + mask_add
    
def mask_bottleneck_in_bias(bias, neuron_idx):     
    with torch.no_grad():
        active_neurons = list(range(neuron_idx))
        train_neuron = active_neurons[-1]
        mask_mul = torch.zeros_like(bias)
        mask_add = torch.zeros_like(bias)
        mask_mul[active_neurons] = 1
        mask_add[train_neuron] = torch.randn(1)
        return bias * mask_mul + mask_add
    
def mask_bottleneck_out_weight(weight, neuron_idx):    
    with torch.no_grad():
        active_neurons = list(range(neuron_idx))
        train_neuron = active_neurons[-1]
        mask_mul = torch.zeros_like(weight)
        mask_add = torch.zeros_like(weight)
        mask_mul[:, active_neurons] = 1                 # weights have shape (latent_dim, encoder[-1])
        mask_add[:, train_neuron] = torch.randn((weight.size(0),))
        return weight * mask_mul + mask_add
    
def mask_bottleneck_out_bias(bias, neuron_idx):     
    with torch.no_grad():
        active_neurons = list(range(neuron_idx))
        train_neuron = active_neurons[-1]
        mask_mul = torch.zeros_like(bias)
        mask_add = torch.zeros_like(bias)
        mask_mul[active_neurons] = 1
        mask_add[train_neuron] = torch.randn(1)
        return bias * mask_mul + mask_add


# ----------------- Functions for hook register -----------------

def mask_bottleneck_in_weight_grad(grad, neuron_idx, freeze_prev_neurons_train=True):     
    mask = torch.zeros_like(grad)
    if freeze_prev_neurons_train:
        mask[neuron_idx-1, :] = 1                   # weights have shape (latent_dim, encoder[-1])
    else:
        active_neurons = list(range(neuron_idx))
        mask[active_neurons, :] = 1                 
    return grad * mask


def mask_bottleneck_in_bias_grad(grad, neuron_idx, freeze_prev_neurons_train=True):    
    mask = torch.zeros_like(grad)
    if freeze_prev_neurons_train:
        mask[neuron_idx-1] = 1
    else:
        active_neurons = list(range(neuron_idx))
        mask[active_neurons] = 1
    return grad * mask


def mask_bottleneck_out_weight_grad(grad, neuron_idx, freeze_prev_neurons_train=True):    
    mask = torch.zeros_like(grad)
    if freeze_prev_neurons_train:
        mask[:, neuron_idx-1] = 1                   # weights have shape (latent_dim, encoder[-1])
    else:
        active_neurons = list(range(neuron_idx))
        mask[:, active_neurons] = 1
    return grad * mask


def mask_bottleneck_out_bias_grad(grad, neuron_idx, freeze_prev_neurons_train=True):  
    mask = torch.zeros_like(grad)
    if freeze_prev_neurons_train:
        mask[neuron_idx-1] = 1
    else:
        active_neurons = list(range(neuron_idx))
        mask[active_neurons] = 1
    return grad * mask



# -------------- TRAIN FUNCTIONS ----------------------



def train_single_neuron(
        model,
        epochs, 
        neuron_to_train, # 1-indexed
        train_loader, 
        val_loader,
        writer,
        lr = 1e-3,
        scheduler = None,
        freeze_prev_neurons_train = True,
        optimizer_func = torch.optim.Adam, 
        save_tensorboard_parameters = False,
        save_tensorboard_bottleneck_parameters = False,
        ):
    
    print(f"\n ------------ Training of neuron{neuron_to_train}---------------")

    # ----------------------------- Optimizer initialization -----------------------------

    optimizer = optimizer_func(model.parameters(), lr=lr, weight_decay=1e-5)

    # ----------------------------- Mask parameters and register hook -----------------------------------

    with torch.no_grad():
        model.bottleneck_in[0].weight.copy_(
            mask_bottleneck_in_weight(model.bottleneck_in[0].weight, neuron_to_train)

            )
        model.bottleneck_in[0].bias.copy_(
            mask_bottleneck_in_bias(model.bottleneck_in[0].bias, neuron_to_train)
        )
        model.bottleneck_out[0].weight.copy_(
            mask_bottleneck_out_weight(model.bottleneck_out[0].weight, neuron_to_train)

            )
        model.bottleneck_out[0].bias.copy_(
            mask_bottleneck_out_bias(model.bottleneck_out[0].bias, neuron_to_train)
        )


    handle_weight_in = model.bottleneck_in[0].weight.register_hook(lambda grad: mask_bottleneck_in_weight_grad(grad, neuron_to_train, freeze_prev_neurons_train))
    handle_bias_in = model.bottleneck_in[0].bias.register_hook(lambda grad: mask_bottleneck_in_bias_grad(grad, neuron_to_train, freeze_prev_neurons_train))
    handle_weight_out = model.bottleneck_out[0].weight.register_hook(lambda grad: mask_bottleneck_out_weight_grad(grad, neuron_to_train, freeze_prev_neurons_train))
    handle_bias_out = model.bottleneck_out[0].bias.register_hook(lambda grad: mask_bottleneck_out_bias_grad(grad, neuron_to_train, freeze_prev_neurons_train))


    #-------------------------- Main loops --------------------------

    global_batch_idx = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # --------------------------- Weight optimization loop --------------------

        for data, _ in train_loader:
            data = data.to(model.device)    
            optimizer.zero_grad()
            output = model(data)
            loss = nn.MSELoss()(output, data)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            global_batch_idx += 1

        if scheduler is not None:
            scheduler.step()

        print(
            "Epoch: {}/{}, Average loss: {:.4f}".format(
                (epoch+1), epochs, train_loss / len(train_loader.dataset)
            )
        )

        # -------------------------- Writer register scalars and parameters --------------------------

        writer.add_scalar(
            "Loss/train", train_loss / len(train_loader.dataset), global_step=(epoch + epochs * neuron_to_train) #to keep trace of the total number of epochs
        )

        model.eval()

        val_loss = 0.0
        with torch.no_grad():
            for data, _ in val_loader:
                data = data.to(model.device)
                output = model(data)
                loss = nn.MSELoss()(output, data)
                val_loss += loss.item()

        writer.add_scalar(
            "Loss/val", val_loss / len(val_loader.dataset), global_step=(epoch + epochs*neuron_to_train)
        )

        if save_tensorboard_bottleneck_parameters:
            for name, param in model.bottleneck_in.named_parameters():
                writer.add_histogram(name, param, global_step=(epoch + epochs*neuron_to_train))
                writer.add_histogram(f"{name}.grad", param.grad, global_step=(epoch + epochs*neuron_to_train))
            for name, param in model.bottleneck_out.named_parameters():
                writer.add_histogram(name, param, global_step=(epoch + epochs*neuron_to_train))
                writer.add_histogram(f"{name}.grad", param.grad, global_step=(epoch + epochs*neuron_to_train))


        if save_tensorboard_parameters:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, global_step=epoch)
                writer.add_histogram(f"{name}.grad", param.grad, global_step=epoch)

    writer.close()

    # ------------------------ Remove hook and zero_grads ----------------------------

    handle_weight_in.remove()
    handle_bias_in.remove()
    handle_weight_out.remove()
    handle_bias_out.remove()

    optimizer.zero_grad()


    print(
        f"Training of neuron {neuron_to_train} completed. Final training loss: {train_loss / len(train_loader.dataset)}, Validation loss: {val_loss / len(val_loader.dataset)}"
    )

    return 0



def train_ProgressiveAE(
        model,
        epochs_for_each_neuron,
        train_loader,
        val_loader,
        writer,
        lr = 1e-3,
        scheduler = None,
        freeze_prev_neurons_train = True,
        optimizer_func = torch.optim.Adam,
        save_tensorboard_parameters = False,
        save_tensorboard_bottleneck_parameters = False,
        ):
    
    print("\n-----------------------------TRAINING STARTED---------------------------- ")

    for neuron_idx in range(1, model.latent_dim + 1):     # neurons are 1-indexed

        train_single_neuron(
            model = model,
            epochs = epochs_for_each_neuron,
            neuron_to_train = neuron_idx,
            train_loader = train_loader,
            val_loader = val_loader,
            writer = writer,
            lr = lr,
            scheduler = scheduler,
            freeze_prev_neurons_train = freeze_prev_neurons_train,
            optimizer_func = optimizer_func,
            save_tensorboard_parameters = save_tensorboard_parameters,
            save_tensorboard_bottleneck_parameters = save_tensorboard_bottleneck_parameters,
            )


    print("------------------------------TRAINING ENDED-------------------------------")










