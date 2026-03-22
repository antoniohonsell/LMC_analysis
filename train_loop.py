import os
import time
import torch


def validate(model, criterion, val_loader, device):
    was_training = model.training
    model.eval()

    val_size = len(val_loader.dataset)
    val_loss_sum = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss_sum += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    if was_training:
        model.train()

    return val_loss_sum / val_size, correct / val_size


def get_train_accuracy(model, train_loader, device):
    was_training = model.training
    model.eval()

    train_size = len(train_loader.dataset)
    correct = 0

    with torch.no_grad():
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

    if was_training:
        model.train()

    return correct / train_size


def train(
    model,
    criterion,
    optimizer,
    scheduler,
    train_loader,
    val_loader,
    epochs,
    device,
    save_dir=None,
    run_name="run",
    save_every=1,
    save_last=True,
    resume_from=None,
    wandb_run=None,
):
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    model = model.to(device)

    train_size = len(train_loader.dataset)

    best_val_loss = float("inf")
    start_epoch = 1

    if resume_from is not None and os.path.isfile(resume_from):
        ckpt = torch.load(resume_from, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        if "history" in ckpt:
            history = ckpt["history"]
        if "best_val_loss" in ckpt:
            best_val_loss = float(ckpt["best_val_loss"])
        # Fast-forward a step-based scheduler (e.g. ScheduledOptimizer) to the
        # right position so its LR curve is not reset.
        if scheduler is None and hasattr(optimizer, "step_num"):
            pass  # step_num already restored via optimizer.load_state_dict
        elif scheduler is not None:
            # Epoch-based schedulers (CosineAnnealingLR): step past completed epochs.
            for _ in range(start_epoch - 1):
                scheduler.step()
        print(f"[resume] Loaded '{resume_from}' — continuing from epoch {start_epoch}/{epochs}")

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        running_loss = 0.0

        print(f"------------------------------\n Epoch: {epoch}")

        t1 = time.time()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        t2 = time.time()

        if scheduler is not None:
                scheduler.step()

        train_loss = running_loss / train_size
        train_acc = get_train_accuracy(model, train_loader, device)
        val_loss, val_acc = validate(model, criterion, val_loader, device)

        # Save best checkpoint (by validation loss)
        if save_dir is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            path = os.path.join(save_dir, f"{run_name}_best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "history": history,
                },
                path,
            )


        print(
            f"time: {int(t2 - t1)}sec "
            f"train_loss: {train_loss:.6f}, train_accuracy: {train_acc:.6f}, "
            f"val_loss: {val_loss:.6f}, val_accuracy: {val_acc:.6f}"
        )

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                },
                step=epoch,
            )

        # periodic checkpoint
        if save_dir is not None and (epoch % save_every == 0):
            path = os.path.join(save_dir, f"{run_name}_epoch{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "history": history,
                },
                path,
            )


    # always save final checkpoint
    if save_dir is not None and save_last:
        path = os.path.join(save_dir, f"{run_name}_final.pth")
        torch.save(
            {
                "epoch": epochs,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "history": history,
            },
            path,
        )

    return history
