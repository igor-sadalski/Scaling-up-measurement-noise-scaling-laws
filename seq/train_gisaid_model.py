import os
import sys
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, EsmForMaskedLM
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
from latentmi import lmi
import pandas as pd

class PreTokenizedDS(Dataset):
    """Dataset for pre-tokenized sequences."""
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, i):
        return {
            'input_ids': self.input_ids[i],
            'attention_mask': self.attention_mask[i]
        }

def train_and_evaluate(model_name, model_id, noise_level, data_path, output_dir):
    """Train model and estimate MI."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load pre-tokenized data
    print(f"Loading data from {data_path}...")
    data = torch.load(data_path, weights_only=False)
    
    train_input_ids = data['train_input_ids']
    train_attention_mask = data['train_attention_mask']
    test_input_ids = data['test_input_ids']
    test_attention_mask = data['test_attention_mask']
    test_seqs = data['test_seqs']
    test_months = data['test_months']
    
    print(f"Train size: {len(train_input_ids)}, Test size: {len(test_input_ids)}")
    
    # Create dataloader
    train_ds = PreTokenizedDS(train_input_ids, train_attention_mask)
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    
    # Initialize tokenizer and model
    tok = AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")
    print(f"Loading model: {model_id}")
    net = EsmForMaskedLM.from_pretrained(model_id).to(device)
    
    # Train
    print("Training...")
    optim = AdamW(net.parameters(), lr=1e-4)
    net.train()
    
    for i, batch in enumerate(tqdm(train_loader, desc="Training")):
        inp = batch['input_ids'].to(device)
        lbl = inp.clone()
        
        # Mask 15% for training
        rand = torch.rand(inp.shape).to(device)
        msk = (rand < 0.15) & (inp != tok.cls_token_id) & (inp != tok.pad_token_id)
        inp[msk] = tok.mask_token_id
        
        loss = net(input_ids=inp, attention_mask=batch['attention_mask'].to(device), labels=lbl).loss
        loss.backward()
        optim.step()
        optim.zero_grad()
    
    # Save model
    model_path = os.path.join(output_dir, f"model_{model_name}_noise_{noise_level:.6f}.pt")
    torch.save(net.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    
    # Extract embeddings
    print("Extracting embeddings...")
    net.eval()
    feats = []
    
    with torch.no_grad():
        # Process in chunks to save memory
        for i in tqdm(range(0, len(test_seqs), 16), desc="Extracting"):
            batch = test_seqs[i:i+16]
            inputs = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            # Use mean pooling of last hidden state
            out = net.base_model(**inputs).last_hidden_state
            
            # Mask out padding tokens for accurate mean
            mask = inputs.attention_mask.unsqueeze(-1).expand(out.size()).float()
            sum_embeddings = torch.sum(out * mask, 1)
            sum_mask = torch.clamp(mask.sum(1), min=1e-9)
            feats.append((sum_embeddings / sum_mask).cpu().numpy())
    
    X = np.vstack(feats)
    
    # Estimate MI
    print("Estimating mutual information...")
    print('number of dimensions: ', X.shape[1])
    pointwise, _, _ = lmi.estimate(X, test_months)
    avg_mi = np.nanmean(pointwise)
    
    print(f"Model: {model_name}, Noise: {noise_level:.6f} -> MI: {avg_mi:.4f}")
    
    # Save results
    result = {
        'model_size': model_name,
        'noise_level': noise_level,
        'mutual_information': avg_mi
    }
    
    result_path = os.path.join(output_dir, f"result_{model_name}_noise_{noise_level:.6f}.csv")
    pd.DataFrame([result]).to_csv(result_path, index=False)
    print(f"Saved results to {result_path}")
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Train GISAID model with specific noise level')
    parser.add_argument('--model-name', type=str, required=True, choices=['8M', '35M', '150M'],
                        help='Model size identifier')
    parser.add_argument('--model-id', type=str, required=True,
                        help='HuggingFace model ID')
    parser.add_argument('--noise-level', type=float, required=True,
                        help='Noise level for this experiment')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to pre-tokenized data file')
    parser.add_argument('--output-dir', type=str, default='seq',
                        help='Output directory for models and results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train and evaluate
    result = train_and_evaluate(
        model_name=args.model_name,
        model_id=args.model_id,
        noise_level=args.noise_level,
        data_path=args.data_path,
        output_dir=args.output_dir
    )
    
    print("\nDone!")

if __name__ == "__main__":
    main()