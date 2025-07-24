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
            "Loss/train", train_loss / len(train_loader.dataset), global_step=epoch
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
            "Loss/val", val_loss / len(val_loader.dataset), global_step=epoch
        )

        if save_tensorboard_parameters:
            for name, param in model.named_parameters():
                writer.add_histogram(name, param, global_step=epoch)
                writer.add_histogram(f"{name}.grad", param.grad, global_step=epoch)

    writer.close()
    print(
        f"Training completed. Final training loss: {train_loss / len(train_loader.dataset)}, Validation loss: {val_loss / len(val_loader.dataset)}"
    )
