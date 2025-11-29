

from argparse import ArgumentParser
from src.loaders import get_parser, get_dataloaders
from medAI.modeling import create_model
from medAI.metrics import calculate_binary_classification_metrics
from medAI.utils.accumulators import DataFrameCollector
import torch
from torch import nn
from tqdm import tqdm
import wandb


def main(): 
    parser = ArgumentParser(parents=[get_parser()])
    # fmt: off
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for the optimizer")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay for the optimizer")
    parser.add_argument("--cls_score", choices=('iou', 'cls'), default='iou',
                        help="Type of classification score to use, either 'iou' or 'cls'")
    parser.set_defaults(
        image_size=224, batch_size=32, num_workers=4
    )
    # fmt: on

    args = parser.parse_args()

    wandb.init(project="prostnfound-img-clf", config=vars(args))

    loaders = get_dataloaders(args)
    
    model = create_model('prostnfound_adapter_medsam_legacy', use_class_decoder=True, prompts=['age', 'psa'])
    model = MetaModel(model, args)
    model.to(args.device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    for epoch in range(args.epochs):
        
        model.train()
        accumulator = DataFrameCollector()
        for data in tqdm(loaders['train'], desc=f"Epoch {epoch + 1}/{args.epochs}"):
            data = model(data)

            loss = data['loss']
            cancer_probs = data['cancer_logits'].sigmoid()
            label = data['label']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            accumulator(dict(cancer_probs=cancer_probs[:, 0], label=label[:, 0]))

        table = accumulator.compute()
        metrics = calculate_binary_classification_metrics(table.cancer_probs, table.label)
        metrics['epoch'] = epoch
        wandb.log(
            {
                f"train/{k}": v for k, v in metrics.items()
            }
        )

        def evaluate(loader, desc): 
            model.eval()
            accumulator = DataFrameCollector()
            with torch.no_grad():
                for data in tqdm(loader, desc=f"{desc} epoch {epoch + 1}/{args.epochs}"):
                    data = model(data)

                    cancer_probs = data['cancer_logits'].sigmoid()
                    label = data['label']
                    accumulator(dict(cancer_probs=cancer_probs[:, 0], label=label[:, 0]))


            table = accumulator.compute()
            metrics = calculate_binary_classification_metrics(table.cancer_probs, table.label)
            metrics['epoch'] = epoch
            wandb.log(
                {
                    f"{desc}/{k}": v for k, v in metrics.items()
                }
            )

        evaluate(loaders['val'], "val")
        evaluate(loaders['test'], "test")

        
class MetaModel(nn.Module):
    def __init__(self, model, args):
        super().__init__()
        self.args = args
        self.model = model
        self.device = args.device

    def forward(self, data):
        x = data['bmode'].to(self.device)
        kwargs = {k: data[k].to(self.device) for k in self.model.prompts}
        model_outputs = self.model(x, **kwargs, output_mode='all')

        if self.args.cls_score == 'iou': 
            cancer_logits = model_outputs['iou']
        elif self.args.cls_score == 'cls':
            cancer_logits = model_outputs['cls_outputs'][0][:, [1]]
        else: 
            raise ValueError(f"Unknown classification score type: {self.args.cls_score}")

        label = data['label'].float().unsqueeze(1).to(self.device)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            cancer_logits, label, reduction='mean'
        )

        data['loss'] = loss
        data['cancer_logits'] = cancer_logits
        data['label'] = label

        return data


if __name__ == "__main__": 
    main()