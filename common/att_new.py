import torch
import torch.nn as nn

class JointOrderAttention(nn.Module):
    def __init__(self, feature_dim):
        super(JointOrderAttention, self).__init__()
        self.feature_dim = feature_dim

        # Linear layer to project the order-specific features to attention scores
        self.attention_projection = nn.Linear(feature_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, gcn_outputs):
        # gcn_outputs is a list of three tensors (first order, second order, third order)
        # Each tensor has shape (batch_size, num_joints, feature_dim)

        # Apply attention to each order
        attended_features = []
        for order_output in gcn_outputs:
            # Reshape to (batch_size * num_joints, feature_dim)
            order_output_reshaped = order_output.view(-1, self.feature_dim)

            # Project the order-specific features to attention scores
            attention_scores = self.attention_projection(order_output_reshaped)

            # Reshape back to (batch_size, num_joints)
            attention_scores = attention_scores.view(-1, order_output.size(1))

            # Apply softmax to get attention weights
            attention_weights = self.softmax(attention_scores)

            # Weighted sum of order-specific features based on attention weights
            order_attended_features = torch.sum(order_output * attention_weights.unsqueeze(-1), dim=1)
            attended_features.append(order_attended_features)

        # Concatenate along the feature dimension
        attended_features = torch.cat(attended_features, dim=-1)

        return attended_features


------------------------------------------------------------------------

import torch.nn.functional as F


class GCNWithAttention(nn.Module):
    def __init__(self, in_channels, out_channels, num_joints):
        super(GCNWithAttention, self).__init__()
        self.gcn_layer = YourGCNLayer(in_channels, out_channels, num_joints)

        # JointOrderAttention module for each order
        self.order_attention_first = JointOrderAttention(out_channels)
        self.order_attention_second = JointOrderAttention(out_channels)
        self.order_attention_third = JointOrderAttention(out_channels)

    def forward(self, x):
        # Assuming x is the input to your GCN layer
        gcn_output_first, gcn_output_second, gcn_output_third = self.gcn_layer(x)

        # Apply attention to select the most relevant order for each joint
        gcn_output_attended_first = self.order_attention_first(gcn_output_first)
        gcn_output_attended_second = self.order_attention_second(gcn_output_second)
        gcn_output_attended_third = self.order_attention_third(gcn_output_third)

        # You can concatenate the attended features or process them further based on your requirements
        final_output = torch.cat([gcn_output_attended_first, gcn_output_attended_second, gcn_output_attended_third], dim=-1)

        return final_output

--------------------------------------------------------------------------------------

import torch.nn.functional as F

class GCNWithAttention(nn.Module):


    def __init__(self, in_channels, out_channels, num_joints):

        super(GCNWithAttention, self).__init__()
        self.gcn_layer = YourGCNLayer(in_channels, out_channels, num_joints)

        # JointOrderAttention module for each order
        self.order_attention_first = JointOrderAttention(out_channels)
        self.order_attention_second = JointOrderAttention(out_channels)
        self.order_attention_third = JointOrderAttention(out_channels)

    def forward(self, x):

        # Assuming x is the input to your GCN layer
        gcn_output_first, gcn_output_second, gcn_output_third = self.gcn_layer(x)

        # Apply attention to select the most relevant order for each joint
        gcn_output_attended_first = self.order_attention_first(gcn_output_first)
        gcn_output_attended_second = self.order_attention_second(gcn_output_second)
        gcn_output_attended_third = self.order_attention_third(gcn_output_third)

        # Add the attended features from different orders
        final_output = gcn_output_attended_first + gcn_output_attended_second + gcn_output_attended_third

        return final_output


----------------------------------------------------



import torch
import torch.nn as nn

class JointOrderAttention(nn.Module):



    def __init__(self, num_orders, features_dim):
        super(JointOrderAttention, self).__init__()
        self.num_orders = num_orders

        # Linear layer to project the order-specific features to attention scores
        self.attention_projection = nn.Linear(features_dim, num_orders)
        self.softmax = nn.Softmax(dim=-1)



    def forward(self, gcn_outputs):
        # gcn_outputs is a tensor of shape (batch_size, num_joints, num_orders, features_dim)

        # Reshape to (batch_size * num_joints, num_orders, features_dim)
        gcn_outputs_reshaped = gcn_outputs.view(-1, self.num_orders, gcn_outputs.size(-1))

        # Project the order-specific features to attention scores
        attention_scores = self.attention_projection(gcn_outputs_reshaped)

        # Reshape back to (batch_size, num_joints, num_orders)
        attention_scores = attention_scores.view(-1, gcn_outputs.size(1), self.num_orders)

        # Apply softmax to get attention weights across orders
        attention_weights = self.softmax(attention_scores)

        # Weighted sum of order-specific features based on attention weights across orders
        attended_features = torch.sum(gcn_outputs * attention_weights.unsqueeze(-1), dim=2)

        return attended_features




------------------------------------------------  cccccccccccccc


import torch
import torch.nn as nn

class JointOrderAttention(nn.Module):
    def __init__(self, features_dim):
        super(JointOrderAttention, self).__init__()

        # Linear layer to project the order-specific features to attention scores
        self.attention_projection = nn.Linear(features_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, gcn_outputs):
        # gcn_outputs is a list of three tensors (first order, second order, third order)
        # Each tensor has shape (batch_size, num_joints, features_dim)

        # Apply attention across orders
        order_attention_scores = []
        for order_output in gcn_outputs:
            # Project the order-specific features to attention scores
            attention_scores = self.attention_projection(order_output)

            # Reshape back to (batch_size, num_joints)
            attention_scores = attention_scores.view(-1, order_output.size(1))

            order_attention_scores.append(attention_scores)

        # Stack attention scores across orders
        order_attention_scores = torch.stack(order_attention_scores, dim=2)

        # Apply softmax to get attention weights across orders
        order_attention_weights = self.softmax(order_attention_scores)

        # Weighted sum of order-specific features based on attention weights across orders
        attended_features = torch.sum(torch.stack(gcn_outputs, dim=2) * order_attention_weights.unsqueeze(-1), dim=2)

        return attended_features





----------------------------------------------------------    ccccccccc jjjjjjjjjjj




import torch
import torch.nn as nn

class CrossJointAttention(nn.Module):
    def __init__(self, num_joints, features_dim):
        super(CrossJointAttention, self).__init__()
        self.num_joints = num_joints
        self.features_dim = features_dim

        # Linear layer to project joint-wise features to attention scores
        self.attention_projection = nn.Linear(features_dim, 1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: (batch_size, num_frames, num_joints * features_dim)

        # Reshape to (batch_size, num_frames, num_joints, features_dim)
        x_reshaped = x.view(-1, x.size(1), self.num_joints, self.features_dim)

        # Project joint-wise features to attention scores
        attention_scores = self.attention_projection(x_reshaped)

        # Reshape back to (batch_size, num_frames, num_joints)
        attention_scores = attention_scores.view(-1, x.size(1), self.num_joints)

        # Apply softmax to get attention weights across joints
        attention_weights = self.softmax(attention_scores)

        # Reshape x to (batch_size, num_frames, num_joints, features_dim) for element-wise multiplication
        x_reshaped = x_reshaped.view(-1, x.size(1), self.num_joints, self.features_dim)

        # Weighted sum of joint-wise features based on attention weights
        attended_features = torch.sum(x_reshaped * attention_weights.unsqueeze(-1), dim=2)

        # Reshape back to (batch_size, num_frames, num_joints * features_dim)
        attended_features = attended_features.view(-1, x.size(1), self.num_joints * self.features_dim)

        return attended_features

# Example usage
batch_size = 16
num_frames = 10
num_joints = 15
features_dim = 64

cross_joint_attention = CrossJointAttention(num_joints, features_dim)

# Create a sample input tensor
input_tensor = torch.randn((batch_size, num_frames, num_joints * features_dim))

# Apply cross-joint attention
output_tensor = cross_joint_attention(input_tensor)

print("Input shape:", input_tensor.shape)
print("Output shape after cross-joint attention:", output_tensor.shape)




------------------------------------------------------------------------------------------

-------------------------------------------------------simple attention

import torch
import torch.nn as nn
import torch.nn.functional as F

class JointSelfAttention(nn.Module):
    def __init__(self, input_dim, num_orders, hidden_dim=128):
        super(JointSelfAttention, self).__init__()

        # Linear layers for query, key, and value projections
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)

        # Linear layer to combine attended values
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, order_features):
        """
        Args:
        - order_features: List of Tensors, each of shape (batch_size, num_joints, features_dim) for each order
        Returns:
        - attended_features: Tensor of shape (batch_size, num_joints, features_dim)
        """

        batch_size, num_joints, features_dim = order_features[0].size()

        # Stack order features along a new dimension
        order_features_stacked = torch.stack(order_features, dim=2)

        # Project queries, keys, and values
        queries = self.query_projection(order_features_stacked)
        keys = self.key_projection(order_features_stacked)
        values = self.value_projection(order_features_stacked)

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(2, 3)) / (features_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=3)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)

        # Linear combination of attended values
        attended_features = self.output_projection(attended_values)

        return attended_features

# Example Usage
input_dim = 64  # Adjust based on the actual dimension of your features
num_orders = 3  # Number of graph order features
hidden_dim = 128  # Dimension of the hidden layers

# Create an instance of JointSelfAttention
joint_self_attention = JointSelfAttention(input_dim, num_orders, hidden_dim)

# Assuming you have three order features: order1, order2, order3
order1 = torch.rand((batch_size, num_joints, input_dim))
order2 = torch.rand((batch_size, num_joints, input_dim))
order3 = torch.rand((batch_size, num_joints, input_dim))

# Apply the JointSelfAttention module
attended_features = joint_self_attention([order1, order2, order3])

# Now, attended_features contains the attended features f



---------------------------------------------  complex attention
import torch
import torch.nn as nn
import torch.nn.functional as F

class NovelJointOrderAttentionWithQV(nn.Module):
    def __init__(self, input_dim, num_orders, hidden_dim=128):
        super(NovelJointOrderAttentionWithQV, self).__init__()

        # Linear layers for query, key, and value projections
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)

        # Linear layers for context vector
        self.context_projection = nn.Linear(input_dim, hidden_dim)

        # Linear layer to combine attended values
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, order_features, context_vector):
        """
        Args:
        - order_features: List of Tensors, each of shape (batch_size, num_joints, features_dim) for each order
        - context_vector: Tensor of shape (batch_size, hidden_dim) representing the context vector
        Returns:
        - attended_features: Tensor of shape (batch_size, num_joints, features_dim)
        """

        batch_size, num_joints, features_dim = order_features[0].size()

        # Stack order features along a new dimension
        order_features_stacked = torch.stack(order_features, dim=2)

        # Project queries, keys, and values
        queries = self.query_projection(order_features_stacked)
        keys = self.key_projection(order_features_stacked)
        values = self.value_projection(order_features_stacked)

        # Project context vector
        context = self.context_projection(context_vector).unsqueeze(1)

        # Incorporate context into queries
        queries = queries + context

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(2, 3)) / (features_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=3)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)

        # Linear combination of attended values
        attended_features = self.output_projection(attended_values)

        return attended_features

# Example Usage
input_dim = 64  # Adjust based on the actual dimension of your features
num_orders = 3  # Number of graph order features
hidden_dim = 128  # Dimension of the hidden layers

# Create an instance of NovelJointOrderAttentionWithQV
novel_joint_order_attention_qv = NovelJointOrderAttentionWithQV(input_dim, num_orders, hidden_dim)

# Assuming you have three order features: order1, order2, order3
order1 = torch.rand((batch_size, num_joints, input_dim))
order2 = torch.rand((batch_size, num_joints, input_dim))
order3 = torch.rand((batch_size, num_joints, input_dim))

# Generate a context vector (you can learn this during training)
context_vector = torch.rand((batch_size, hidden_dim))

# Apply the NovelJointOrderAttentionWithQV module
attended_features = novel_joint_order_attention_qv([order1, order2, order3], context_vector)

# Now, attended_features contains the attended features for each joint

--------------------------------------------------------------------------------------------------------------------------  cross order


class CrossOrderAttentionWithQV(nn.Module):
    def __init__(self, input_dim, num_orders, hidden_dim=128):
        super(CrossOrderAttentionWithQV, self).__init__()

        # Linear layers for query, key, and value projections
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)

        # Linear layers for context vector
        self.context_projection = nn.Linear(input_dim, hidden_dim)

        # Linear layer to combine attended values
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, order_features, context_vector):
        """
        Args:
        - order_features: List of Tensors, each of shape (batch_size, num_joints, features_dim) for each order
        - context_vector: Tensor of shape (batch_size, hidden_dim) representing the context vector
        Returns:
        - attended_features: Tensor of shape (batch_size, num_joints, features_dim)
        """

        batch_size, num_joints, features_dim = order_features[0].size()

        # Stack order features along a new dimension
        order_features_stacked = torch.stack(order_features, dim=2)

        # Project queries, keys, and values
        queries = self.query_projection(order_features_stacked)
        keys = self.key_projection(order_features_stacked)
        values = self.value_projection(order_features_stacked)

        # Project context vector
        context = self.context_projection(context_vector).unsqueeze(1)

        # Incorporate context into queries
        queries = queries + context

        # Compute attention scores for each joint and order
        attention_scores = torch.matmul(queries, keys.transpose(2, 3)) / (features_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=3)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)

        # Linear combination of attended values
        attended_features = self.output_projection(attended_values)

        return attended_features

# Example Usage
input_dim = 64  # Adjust based on the actual dimension of your features
num_orders = 3  # Number of graph order features
hidden_dim = 128  # Dimension of the hidden layers

# Create an instance of CrossOrderAttentionWithQV
cross_order_attention_qv = CrossOrderAttentionWithQV(input_dim, num_orders, hidden_dim)

# Assuming you have three order features: order1, order2, order3
order1 = torch.rand((batch_size, num_joints, input_dim))
order2 = torch.rand((batch_size, num_joints, input_dim))
order3 = torch.rand((batch_size, num_joints, input_dim))

# Generate a context vector (you can learn this during training)
context_vector = torch.rand((batch_size, hidden_dim))

# Apply the CrossOrderAttentionWithQV module
attended_features = cross_order_attention_qv([order1, order2, order3], context_vector)

# Now, attended_features contains the attended features for each joint

-----------------------------------------------------------------------------------------------------------------------------  cross joints att qkv

import torch
import torch.nn as nn
import torch.nn.functional as F



class TemporalJointAttention(nn.Module):
    def __init__(self, input_dim, num_joints, hidden_dim=128):
        super(TemporalJointAttention, self).__init__()

        # Linear layers for query, key, and value projections
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)

        # Linear layer to combine attended values
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, input_features):
        """
        Args:
        - input_features: Tensor of shape (batch_size, seq_length, num_joints, features_dim) for skeleton sequences
        Returns:
        - attended_features: Tensor of shape (batch_size, seq_length, num_joints, features_dim)
        """

        batch_size, seq_length, num_joints, features_dim = input_features.size()

        # Reshape input features for attention computation
        reshaped_input = input_features.view(batch_size * seq_length, num_joints, features_dim)

        # Project queries, keys, and values
        queries = self.query_projection(reshaped_input)
        keys = self.key_projection(reshaped_input)
        values = self.value_projection(reshaped_input)

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(1, 2)) / (features_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=2)

        # Apply attention to values
        attended_values = torch.matmul(attention_weights, values)

        # Reshape back to the original shape
        attended_values = attended_values.view(batch_size, seq_length, num_joints, features_dim)

        # Linear combination of attended values
        attended_features = self.output_projection(attended_values)

        return attended_features

# Example Usage
input_dim = 64  # Adjust based on the actual dimension of your features
num_joints = 17  # Number of joints in the skeleton
hidden_dim = 128  # Dimension of the hidden layers

# Create an instance of TemporalJointAttention
temporal_joint_attention = TemporalJointAttention(input_dim, num_joints, hidden_dim)

# Assuming you have skeleton sequences as input features
skeleton_sequences = torch.rand((batch_size, seq_length, num_joints, input_dim))

# Apply the TemporalJointAttention module
attended_skeleton_sequences = temporal_joint_attention(skeleton_sequences)

# Now, attended_skeleton_sequences contains the attended features across frames for each joint


------------------------------------------------------------------------------------------------------------------------------------------ cross joints attention lstm



import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexTemporalJointAttention(nn.Module):
    def __init__(self, input_dim, num_joints, hidden_dim=128, num_layers=2):
        super(ComplexTemporalJointAttention, self).__init__()

        # 1D Convolutional layers for joint interactions
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1)
            for _ in range(num_layers)
        ])

        # Recurrent layer for temporal dependencies
        self.recurrent_layer = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)

        # Linear layer for output
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, input_features):
        """
        Args:
        - input_features: Tensor of shape (batch_size, seq_length, num_joints, features_dim) for skeleton sequences
        Returns:
        - attended_features: Tensor of shape (batch_size, seq_length, num_joints, features_dim)
        """

        batch_size, seq_length, num_joints, features_dim = input_features.size()

        # Reshape input features for convolutional operation
        reshaped_input = input_features.view(batch_size, seq_length * num_joints, features_dim).permute(0, 2, 1)

        # Apply 1D convolutional layers for joint interactions
        for conv_layer in self.conv_layers:
            reshaped_input = F.relu(conv_layer(reshaped_input))

        # Reshape back to the original shape
        reshaped_input = reshaped_input.permute(0, 2, 1).view(batch_size, seq_length, num_joints, -1)

        # Apply recurrent layer for capturing temporal dependencies
        recurrent_output, _ = self.recurrent_layer(reshaped_input)

        # Apply linear layer for output projection
        attended_features = self.output_projection(recurrent_output)

        return attended_features

# Example Usage
input_dim = 64  # Adjust based on the actual dimension of your features
num_joints = 17  # Number of joints in the skeleton
hidden_dim = 128  # Dimension of the hidden layers
num_layers = 2  # Number of convolutional layers

# Create an instance of ComplexTemporalJointAttention
complex_temporal_joint_attention = ComplexTemporalJointAttention(input_dim, num_joints, hidden_dim, num_layers)

# Assuming you have skeleton sequences as input features
skeleton_sequences = torch.rand((batch_size, seq_length, num_joints, input_dim))

# Apply the ComplexTemporalJointAttention module
attended_skeleton_sequences = complex_temporal_joint_attention(skeleton_sequences)

# Now, attended_skeleton_sequences contain

---------------------------------------------------------------------------------- cross joints  trans encoder


import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexTemporalJointAttention(nn.Module):
    def __init__(self, input_dim, num_joints, hidden_dim=128, num_layers=2):
        super(ComplexTemporalJointAttention, self).__init__()

        # 1D Convolutional layers for joint interactions
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=1)
            for _ in range(num_layers)
        ])

        # Transformer layer for temporal dependencies
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,  # Adjust nhead based on your specific requirements
            dim_feedforward=hidden_dim * 4
        )
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=num_layers)

        # Linear layer for output
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, input_features):
        """
        Args:
        - input_features: Tensor of shape (batch_size, seq_length, num_joints, features_dim) for skeleton sequences
        Returns:
        - attended_features: Tensor of shape (batch_size, seq_length, num_joints, features_dim)
        """

        batch_size, seq_length, num_joints, features_dim = input_features.size()

        # Reshape input features for convolutional operation
        reshaped_input = input_features.view(batch_size, seq_length * num_joints, features_dim).permute(0, 2, 1)

        # Apply 1D convolutional layers for joint interactions
        for conv_layer in self.conv_layers:
            reshaped_input = F.relu(conv_layer(reshaped_input))

        # Reshape back to the original shape
        reshaped_input = reshaped_input.permute(0, 2, 1).view(batch_size, seq_length, num_joints, -1)

        # Reshape input for transformer
        transformer_input = reshaped_input.permute(1, 0, 2, 3).contiguous()

        # Apply transformer layer for capturing temporal dependencies
        transformer_output = self.transformer_encoder(transformer_input)

        # Reshape back to the original shape
        transformer_output = transformer_output.permute(1, 0, 2, 3).contiguous()

        # Apply linear layer for output projection
        attended_features = self.output_projection(transformer_output)

        return attended_features

# Example Usage
input_dim = 64  # Adjust based on the actual dimension of your features
num_joints = 17  # Number of joints in the skeleton
hidden_dim = 128  # Dimension of the hidden layers
num_layers = 2  # Number of convolutional and transformer layers

# Create an instance of ComplexTemporalJointAttention
complex_temporal_joint_attention = ComplexTemporalJointAttention(input_dim, num_joints, hidden_dim, num_layers)

# Assuming you have skeleton sequences as input features
skeleton_sequences = torch.rand((batch_size, seq_length, num_joints, input_dim))

# Apply the ComplexTemporalJointAttention module
attended_skeleton_sequences = complex_temporal_joint_attention(skeleton_sequences)

# Now, attended_skeleton_sequences contains the attended ith complex temporal joint attention using transformers

-------------------------------------------------------------------------------------------------------------------------------------------------------------  TCN

import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexTemporalJointAttention(nn.Module):
    def __init__(self, input_dim, num_joints, hidden_dim=128, num_layers=2, kernel_size=3):
        super(ComplexTemporalJointAttention, self).__init__()

        # 1D Convolutional layers for joint interactions
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
            for _ in range(num_layers)
        ])

        # Linear layer for output
        self.output_projection = nn.Linear(hidden_dim, input_dim)

    def forward(self, input_features):
        """
        Args:
        - input_features: Tensor of shape (batch_size, seq_length, num_joints, features_dim) for skeleton sequences
        Returns:
        - attended_features: Tensor of shape (batch_size, seq_length, num_joints, features_dim)
        """

        batch_size, seq_length, num_joints, features_dim = input_features.size()

        # Reshape input features for convolutional operation
        reshaped_input = input_features.view(batch_size * seq_length, num_joints, features_dim).permute(0, 2, 1)

        # Apply 1D convolutional layers for joint interactions
        for conv_layer in self.conv_layers:
            reshaped_input = F.relu(conv_layer(reshaped_input))

        # Reshape back to the original shape
        reshaped_input = reshaped_input.permute(0, 2, 1).view(batch_size, seq_length, num_joints, -1)

        # Apply linear layer for output projection
        attended_features = self.output_projection(reshaped_input)

        return attended_features

# Example Usage
input_dim = 64  # Adjust based on the actual dimension of your features
num_joints = 17  # Number of joints in the skeleton
hidden_dim = 128  # Dimension of the hidden layers
num_layers = 2  # Number of convolutional layers
kernel_size = 3  # Size of the convolutional kernel

# Create an instance of ComplexTemporalJointAttention
complex_temporal_joint_attention = ComplexTemporalJointAttention(input_dim, num_joints, hidden_dim, num_layers, kernel_size)

# Assuming you have skeleton sequences as input features
skeleton_sequences = torch.rand((batch_size, seq_length, num_joints, input_dim))

# Apply the ComplexTemporalJointAttention module
attended_skeleton_sequences = complex_temporal_joint_attention(skeleton_sequences)

# Now, attended_skeleton_sequences contains the attended features with complex temporal joint attention using temporal convolution


---------------------------------------------------------------------------   attention gcn each order


import torch
import torch.nn as nn
import torch.nn.functional as F

class HGraphConv(nn.Module):
    """ 
    High-order graph convolution layer 
    """

    def __init__(self, in_features, out_features, adj, bias=True):

        super(HGraphConv, self).__init__()

        self.adj = adj

        # Linear layer to project the order-specific features to attention scores
        self.attention_projection = nn.Linear(out_features, 1)
        self.softmax = nn.Softmax(dim=1)

        if self.adj.size(0) == 17:
            self.in_features = in_features
            self.out_features = out_features

            self.W = nn.Parameter(torch.zeros(size=(4, in_features, out_features), dtype=torch.float))
            nn.init.xavier_uniform_(self.W.data, gain=1.414)

            self.adj_0 = torch.eye(adj.size(0), dtype=torch.float)  # declare self-connections
            self.m_0 = (self.adj_0 > 0)
            self.e_0 = nn.Parameter(torch.zeros(1, len(self.m_0.nonzero()), dtype=torch.float))
            nn.init.constant_(self.e_0.data, 1)

            self.adj_1 = adj  # one_hop neighbors
            self.m_1 = (self.adj_1 > 0)
            self.e_1 = nn.Parameter(torch.zeros(1, len(self.m_1.nonzero()), dtype=torch.float))
            nn.init.constant_(self.e_1.data, 1)

            self.adj_2 = torch.matmul(self.adj_1, adj)  # two_hop neighbors
            self.m_2 = (self.adj_2 > 0)
            self.e_2 = nn.Parameter(torch.zeros(1, len(self.m_2.nonzero()), dtype=torch.float))
            nn.init.constant_(self.e_2.data, 1)

            self.adj_3 = torch.matmul(self.adj_2, adj)  # three_hop neighbors
            self.m_3 = (self.adj_3 > 0)
            self.e_3 = nn.Parameter(torch.zeros(1, len(self.m_3.nonzero()), dtype=torch.float))
            nn.init.constant_(self.e_3.data, 1)

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
                stdv = 1. / math.sqrt(self.W.size(2))
                self.bias.data.uniform_(-stdv, stdv)

                self.bias_2 = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
                stdv = 1. / math.sqrt(self.W.size(2))
                self.bias_2.data.uniform_(-stdv, stdv)
            else:
                self.register_parameter('bias', None)

    def forward(self, input):

        if self.adj.size(0) == 17:
            h0 = torch.matmul(input, self.W[0])
            h1 = torch.matmul(input, self.W[1])
            h2 = torch.matmul(input, self.W[2])
            h3 = torch.matmul(input, self.W[3])

            # Simple self-attention mechanism
            att_0 = self.softmax(self.attention_projection(h0).squeeze(dim=-1))
            att_1 = self.softmax(self.attention_projection(h1).squeeze(dim=-1))
            att_2 = self.softmax(self.attention_projection(h2).squeeze(dim=-1))
            att_3 = self.softmax(self.attention_projection(h3).squeeze(dim=-1))

            # Apply attention to features
            h0 = torch.matmul(att_0.unsqueeze(dim=1), h0)
            h1 = torch.matmul(att_1.unsqueeze(dim=1), h1)
            h2 = torch.matmul(att_2.unsqueeze(dim=1), h2)
            h3 = torch.matmul(att_3.unsqueeze(dim=1), h3)

            A_0 = -9e15 * torch.ones_like(self.adj_0).to(input.device)
            A_1 = -9e15 * torch.ones_like(self.adj_1).to(input.device)  # without self-connection
            A_2 = -9e15 * torch.ones_like(self.adj_2).to(input.device)
            A_3 = -9e15 * torch.ones_like(self.adj_3).to(input.device)

            A_0[self.m_0] = self.e_0
            A_1[self.m_1] = self.e_1
            A_2[self.m_2] = self.e_2
            A_3[self.m_3] = self.e_3

            A_0 = F.softmax(A_0, dim=1)
            A_1 = F.softmax(A_1, dim=1)
            A_2 = F.softmax(A_2, dim=1)
            A_3 = F.softmax(A_3, dim=1)

            output_0 = torch.matmul(A_0, h0)
            output_1 = torch.matmul(A_1, h1)
            output_2 = torch.matmul(A_2, h2)
            output_3 = torch.matmul(A_3, h3)

            # Combine the outputs
            output = output_0 + output_1 + output_2 + output_3

            # Add biases if present
            if hasattr(self, 'bias'):
                output += self.bias
                output += self.bias_2

            return output

------------------------------------------------------------------------ gcn attention 2


import torch
import torch.nn as nn
import torch.nn.functional as F

class HGraphConv(nn.Module):
    """ 
    High-order graph convolution layer 
    """

    def __init__(self, in_features, out_features, adj, bias=True):

        super(HGraphConv, self).__init__()

        self.adj = adj

        # Non-linear layer to modulate attention scores
        self.attention_modulation = nn.Sequential(
            nn.Linear(out_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, 1)
        )
        self.softmax = nn.Softmax(dim=1)

        if self.adj.size(0) == 17:
            self.in_features = in_features
            self.out_features = out_features

            self.W = nn.Parameter(torch.zeros(size=(4, in_features, out_features), dtype=torch.float))
            nn.init.xavier_uniform_(self.W.data, gain=1.414)

            self.adj_0 = torch.eye(adj.size(0), dtype=torch.float)  # declare self-connections
            self.m_0 = (self.adj_0 > 0)
            self.e_0 = nn.Parameter(torch.zeros(1, len(self.m_0.nonzero()), dtype=torch.float))
            nn.init.constant_(self.e_0.data, 1)

            self.adj_1 = adj  # one_hop neighbors
            self.m_1 = (self.adj_1 > 0)
            self.e_1 = nn.Parameter(torch.zeros(1, len(self.m_1.nonzero()), dtype=torch.float))
            nn.init.constant_(self.e_1.data, 1)

            self.adj_2 = torch.matmul(self.adj_1, adj)  # two_hop neighbors
            self.m_2 = (self.adj_2 > 0)
            self.e_2 = nn.Parameter(torch.zeros(1, len(self.m_2.nonzero()), dtype=torch.float))
            nn.init.constant_(self.e_2.data, 1)

            self.adj_3 = torch.matmul(self.adj_2, adj)  # three_hop neighbors
            self.m_3 = (self.adj_3 > 0)
            self.e_3 = nn.Parameter(torch.zeros(1, len(self.m_3.nonzero()), dtype=torch.float))
            nn.init.constant_(self.e_3.data, 1)

            if bias:
                self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
                stdv = 1. / math.sqrt(self.W.size(2))
                self.bias.data.uniform_(-stdv, stdv)

                self.bias_2 = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
                stdv = 1. / math.sqrt(self.W.size(2))
                self.bias_2.data.uniform_(-stdv, stdv)
            else:
                self.register_parameter('bias', None)

    def forward(self, input):

        if self.adj.size(0) == 17:
            h0 = torch.matmul(input, self.W[0])
            h1 = torch.matmul(input, self.W[1])
            h2 = torch.matmul(input, self.W[2])
            h3 = torch.matmul(input, self.W[3])

            # Novel attention modulation
            modulated_att_0 = self.attention_modulation(h0)
            modulated_att_1 = self.attention_modulation(h1)
            modulated_att_2 = self.attention_modulation(h2)
            modulated_att_3 = self.attention_modulation(h3)

            att_0 = self.softmax(modulated_att_0.squeeze(dim=-1))
            att_1 = self.softmax(modulated_att_1.squeeze(dim=-1))
            att_2 = self.softmax(modulated_att_2.squeeze(dim=-1))
            att_3 = self.softmax(modulated_att_3.squeeze(dim=-1))

            # Apply attention to features
            h0 = torch.matmul(att_0.unsqueeze(dim=1), h0)
            h1 = torch.matmul(att_1.unsqueeze(dim=1), h1)
            h2 = torch.matmul(att_2.unsqueeze(dim=1), h2)
            h3 = torch.matmul(att_3.unsqueeze(dim=1), h3)

            A_0 = -9e15 * torch.ones_like(self.adj_0).to(input.device)
            A_1 = -9e15 * torch.ones_like(self.adj_1).to(input.device)  # without self-connection
            A_2 = -9e15 * torch.ones_like(self.adj_2).to(input.device)
            A_3 = -9e15 * torch.ones_like(self.adj_3).to(input.device)

            A_0[self.m_0] = self.e_0
            A_1


-----------------------------------------------------

Paper Outline:
Title: Enhancing 2D to 3D Pose Estimation with OrderSelectorAttention Mechanism

Abstract:

Briefly summarize the proposed OrderSelectorAttention mechanism and its impact on 2D to 3D pose estimation.
Introduction:

Introduce the problem of 2D to 3D pose estimation.
Highlight the importance of attention mechanisms in capturing relevant information.
Provide an overview of the proposed OrderSelectorAttention mechanism.
Related Work:

Discuss existing attention mechanisms used in pose estimation.
Highlight the limitations and challenges in existing approaches.
Emphasize the need for a novel attention mechanism like OrderSelectorAttention.
Methodology:

Explain the architecture of the 2D to 3D pose estimation model with GCN and Transformer.
Introduce the OrderSelectorAttention mechanism and its integration into the model.
Experimental Setup:

Describe datasets used for evaluation.
Specify model parameters and training details.
Results:

Present quantitative results comparing the proposed model with and without OrderSelectorAttention.
Visualize attention maps to show how the mechanism selects relevant order features.
Discussion:

Analyze the experimental results.
Discuss the advantages of OrderSelectorAttention in capturing the most informative order for each joint.
Conclusion:

Summarize the key contributions.
Discuss potential future directions for improving the proposed mechanism.
This outline provides a structure for a paper that introduces the OrderSelectorAttention mechanism and evaluates its effectiveness in enhancing 2D to 3D pose estimation.













------------------------cross order new   (two below)

import torch
import torch.nn as nn
import torch.nn.functional as F

class JointAttentionCrossOrdersPerJoint(nn.Module):
    def __init__(self, input_dim, num_orders, hidden_dim=128):
        super(JointAttentionCrossOrdersPerJoint, self).__init__()

        # Linear layers for query, key, value, and attention projections
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)
        self.attention_projection = nn.Linear(hidden_dim, 1)

        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=1)

    def forward(self, gcn_outs):
        """
        Args:
        - gcn_outs: List of Tensors, each of shape (batch_size, num_joints, features_dim) for each order
        Returns:
        - attended_features: Tensor of shape (batch_size, num_joints, features_dim)
        """

        batch_size, num_joints, features_dim = gcn_outs[0].size()

        # Stack order features along a new dimension
        joint_features_stacked = torch.stack(gcn_outs, dim=2)

        # Project joint-specific features to query and key
        queries = self.query_projection(joint_features_stacked)
        keys = self.key_projection(joint_features_stacked)

        # Compute attention scores
        attention_scores = self.attention_projection(torch.tanh(queries + keys))

        # Reshape back to (batch_size, num_joints, num_orders)
        attention_scores = attention_scores.view(batch_size, num_joints, num_orders)

        # Apply softmax to get attention weights across orders
        joint_attention_weights = self.softmax(attention_scores)

        # Weighted sum of joint-specific features based on attention weights across orders
        attended_features = torch.sum(joint_features_stacked * joint_attention_weights.unsqueeze(-1), dim=2)

        return attended_features


-----------------------------------------------------------------



import torch
import torch.nn as nn
import torch.nn.functional as F

class JointAttentionCrossOrdersWithQK(nn.Module):
    def __init__(self, input_dim, num_orders, hidden_dim=128):
        super(JointAttentionCrossOrdersWithQK, self).__init__()

        # Linear layers for query, key, value, and attention projections
        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)
        self.attention_projection = nn.Linear(hidden_dim, 1)

        # Softmax for attention weights
        self.softmax = nn.Softmax(dim=2)

    def forward(self, gcn_outs):
        """
        Args:
        - gcn_outs: List of Tensors, each of shape (batch_size, num_joints, features_dim) for each order
        Returns:
        - attended_features: Tensor of shape (batch_size, num_joints, features_dim)
        """

        order_attention_scores = []

        for order_output in gcn_outs:
            # Project order-specific features to query and key
            queries = self.query_projection(order_output)
            keys = self.key_projection(order_output)

            # Compute attention scores
            attention_scores = self.attention_projection(torch.tanh(queries + keys))

            # Reshape back to (batch_size, num_joints)
            attention_scores = attention_scores.view(-1, order_output.size(1))

            order_attention_scores.append(attention_scores)

        # Stack attention scores across orders
        order_attention_scores = torch.stack(order_attention_scores, dim=2)

        # Apply softmax to get attention weights across orders
        order_attention_weights = self.softmax(order_attention_scores)

        # Weighted sum of order-specific features based on attention weights across orders
        attended_features = torch.sum(torch.stack(gcn_outs, dim=2) * order_attention_weights.unsqueeze(-1), dim=2)

        return attended_features


----------------------------------------------------------------------



import torch.nn as nn
from transformer.Models import TransformerEncoder  # Assuming you have a TransformerEncoder module

class ComplexAttentionProjection(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_heads, output_dim):
        super(ComplexAttentionProjection, self).__init__()

        # Convolutional layers for spatial relations
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, output_dim, kernel_size=3, padding=1),
            nn.ReLU()
        ])

        # Transformer layer for capturing temporal relations
        self.transformer_encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=output_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim
        )

    def forward(self, x):
        # Reshape input for convolutional layers
        conv_input = x.permute(0, 3, 1, 2).contiguous().view(-1, x.size(1), x.size(2))

        # Apply convolutional layers
        for layer in self.conv_layers:
            conv_input = layer(conv_input)

        # Reshape back to the original shape
        conv_output = conv_input.view(x.size(0), x.size(3), x.size(1), x.size(2)).permute(0, 2, 3, 1).contiguous()

        # Apply Transformer layer for capturing temporal relations
        transformer_input = conv_output.view(-1, x.size(1), x.size(2) * x.size(3))
        transformer_output = self.transformer_encoder(transformer_input)

        # Reshape back to the original shape
        transformer_output = transformer_output.view(x.size(0), x.size(2), x.size(3), x.size(1)).permute(0, 3, 1, 2).contiguous()

        return transformer_output

# Usage:
input_dim = p * 32  # Assuming p and 32 are your input dimensions
hidden_dim = 64
num_layers = 2
num_heads = 4
output_dim = p  # Output dimension should match the number of joints

complex_attention_projection = ComplexAttentionProjection(input_dim, hidden_dim, num_layers, num_heads, output_dim)
























