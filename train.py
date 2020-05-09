from model import SigNet, ContrastiveLoss
from data import get_data_loader, Binarize
from PIL import ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

seed = 2020
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device: {}'.format(device))

def train(model, optimizer, criterion, dataloader, log_interval=50):
    model.train()
    running_loss = 0
    number_samples = 0

    for batch_idx, (x1, x2, y) in enumerate(dataloader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        optimizer.zero_grad()
        x1, x2 = model(x1, x2)
        loss = criterion(x1, x2, y)
        loss.backward()
        optimizer.step()

        number_samples += len(x1)
        running_loss += loss.item() * len(x1)
        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx+1, len(dataloader), running_loss / number_samples))
            running_loss = 0
            number_samples = 0

@torch.no_grad()
def eval(model, criterion, dataloader, log_interval=50):
    model.eval()
    running_loss = 0
    number_samples = 0

    for batch_idx, (x1, x2, y) in enumerate(dataloader):
        x1, x2, y = x1.to(device), x2.to(device), y.to(device)

        x1, x2 = model(x1, x2)
        loss = criterion(x1, x2, y)

        number_samples += len(x1)
        running_loss += loss.item() * len(x1)

        if (batch_idx + 1) % log_interval == 0 or batch_idx == len(dataloader) - 1:
            print('{}/{}: Loss: {:.4f}'.format(batch_idx+1, len(dataloader), running_loss / number_samples))
    return running_loss / number_samples

if __name__ == "__main__":
    model = SigNet().to(device)
    criterion = ContrastiveLoss(alpha=1, beta=1, margin=1).to(device)
    optimizer = optim.RMSprop(model.parameters(), lr=1e-5, eps=1e-8, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.1)
    num_epochs = 20

    image_transform = transforms.Compose([
        transforms.Resize((155, 220)),
        ImageOps.invert,
        transforms.ToTensor(),
        # TODO: add normalize
    ])

    trainloader = get_data_loader(is_train=True, batch_size=24, image_transform=image_transform, dataset='cedar')
    testloader = get_data_loader(is_train=False, batch_size=24, image_transform=image_transform, dataset='cedar')

    model.train()
    print(model)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('Training', '-'*20)
        train(model, optimizer, criterion, trainloader)
        print('Evaluating', '-'*20)
        loss_pe = eval(model, criterion, testloader)
        scheduler.step(epoch)

        to_save = {
            'model': model.module.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optim': optimizer.state_dict(),
        }

        print('Saving checkpoint..')
        torch.save(to_save, 'checkpoints/epoch_{}_loss_{:.3f}.pt'.format(epoch, loss_pe))

    print('Done')
