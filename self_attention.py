import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Self-Attention Model for Trajectory Prediction
class TrajectorySelfAttention(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_heads=4, future_steps=5, max_humans=5):
        super(TrajectorySelfAttention, self).__init__()
        self.input_dim = input_dim  # (x, y) coordinates
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.future_steps = future_steps
        self.max_humans = max_humans

        # Embedding layer for input positions
        self.embedding = nn.Linear(input_dim, hidden_dim)

        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        # Output layer to predict future steps
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim * future_steps)  # Predict future_steps * (x, y)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len=5, num_humans, input_dim=2]
        batch_size, seq_len, num_humans, _ = x.size()

        # Reshape for attention: [batch_size * num_humans, seq_len, input_dim]
        x = x.view(batch_size * num_humans, seq_len, self.input_dim)

        # Embed input
        x_embed = self.embedding(x)  # [batch_size * num_humans, seq_len, hidden_dim]

        # Apply self-attention
        attn_output, _ = self.self_attention(x_embed, x_embed, x_embed)  # [batch_size * num_humans, seq_len, hidden_dim]

        # Use the last time step’s output for prediction
        attn_output = attn_output[:, -1, :]  # [batch_size * num_humans, hidden_dim]

        # Predict future steps
        pred = self.fc(attn_output)  # [batch_size * num_humans, future_steps * input_dim]
        pred = pred.view(batch_size, num_humans, self.future_steps, self.input_dim)  # [batch_size, num_humans, future_steps, input_dim]

        return pred

# Training function
def train_model(model, data, labels, num_epochs=50, batch_size=32):
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    num_samples = len(data)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range(0, num_samples, batch_size):
            batch_data = data[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]

            optimizer.zero_grad()
            pred = model(batch_data)  # [batch_size, num_humans, future_steps, 2]
            loss = criterion(pred, batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / (num_samples // batch_size)}")

    # Save the trained model
    torch.save(model.state_dict(), "trajectory_model.pth")
    print("Model trained and saved to trajectory_model.pth")