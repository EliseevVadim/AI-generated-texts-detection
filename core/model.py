import torch
from torch.nn.utils import clip_grad_norm_

from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score, f1_score

from core.utils import free_memory


def get_model_predictions(model, data_loader, device='cpu'):
    ground_truth = []
    predictions = []
    probabilities = []

    model.eval()
    with torch.no_grad():
        for input_ids, attention_mask, labels in data_loader:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = torch.mean(outputs.logits, dim=1)
            probs = torch.softmax(logits, dim=1)[:, 1]
            _, indices = torch.max(logits, 1)

            predictions.extend(indices.tolist())
            probabilities.extend(probs.tolist())
            ground_truth.extend(labels.tolist())

            del input_ids
            del attention_mask
            del labels
            del outputs
            del logits
            del indices
            del probs
            free_memory()
    return ground_truth, predictions, probabilities


def train_model(model, train_loader, criterion, optimizer, num_epochs,
                step_size=1, gamma=0.5, device='cpu'):
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_history = []
    train_history = []
    f1_history = []

    for epoch in range(num_epochs):
        model.train()

        loss_accum = 0

        all_labels = []
        all_predictions = []

        for i_step, (input_ids, attention_mask, y) in enumerate(train_loader):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = torch.mean(outputs.logits, dim=1)

            optimizer.zero_grad()
            loss = criterion(logits, y)
            loss.backward()
            clip_grad_norm_(model.lm_head.parameters(), 1.0)
            optimizer.step()

            _, indices = torch.max(logits, 1)

            loss_accum += loss

            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(indices.cpu().numpy())

            del outputs
            del logits
            del indices
            del input_ids
            del attention_mask
            del y
            free_memory()

        scheduler.step()
        ave_loss = loss_accum / i_step
        train_accuracy = accuracy_score(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions)

        loss_history.append(float(ave_loss))
        train_history.append(train_accuracy)
        f1_history.append(f1)

        free_memory()

        print(f"Average loss: {ave_loss}, Train accuracy: {train_accuracy}, Train f1 score: {f1}")

    return loss_history, train_history, f1_history
