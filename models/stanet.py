import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# If you have any questions, please contact us at shulmt@shu.edu.cn

def pearson_r(eeg, fnirs):
    """
    Calculates the Pearson correlation coefficient between EEG and fNIRS tensors.
    Args:
        eeg (torch.Tensor): The EEG tensor.
        fnirs (torch.Tensor): The fNIRS tensor.
    Returns:
        torch.Tensor: The mean Pearson correlation coefficient across the batch.
    """
    # Ensure tensors are on the same device and dtype
    fnirs = fnirs.to(eeg.device, eeg.dtype)
    
    # Calculate means along the feature dimension (dim=1)
    mx = torch.mean(eeg, dim=1, keepdim=True)
    my = torch.mean(fnirs, dim=1, keepdim=True)
    
    # Center the tensors
    xm, ym = eeg - mx, fnirs - my
    
    # Calculate the numerator of the Pearson correlation
    r_num = torch.mean(xm * ym, dim=1)
    
    # Calculate the denominator of the Pearson correlation
    # Use torch.std with correction=0 for population standard deviation
    r_den = torch.std(xm, dim=1, unbiased=False) * torch.std(ym, dim=1, unbiased=False) + 1e-6
    
    # Calculate Pearson correlation coefficient
    plcc = r_num / r_den
    plcc = torch.abs(plcc)
    
    # Calculate the mean across the batch
    plcc_meanbatch = torch.mean(plcc)
    
    return plcc_meanbatch


class PosEmbedding(nn.Module):
    """
    A layer to add positional embeddings to the input.
    """
    def __init__(self, input_shape):
        super(PosEmbedding, self).__init__()
        # Create a trainable parameter for positional embeddings
        self.pos_embedding = nn.Parameter(torch.empty(1, input_shape[-2], input_shape[-1]))
        # Initialize the parameter using He uniform initialization
        nn.init.kaiming_uniform_(self.pos_embedding, a=np.sqrt(5))

    def forward(self, inputs):
        """
        Adds the positional embedding to the input tensor.
        """
        return inputs + self.pos_embedding


class EFAttention(nn.Module):
    """
    EEG-fNIRS Attention layer. This layer computes attention between EEG and fNIRS
    signals and calculates a correlation-based loss.
    """
    def __init__(self, emb_size, d_model, heads, drop):
        super(EFAttention, self).__init__()
        # Note: In PyTorch, d_model (embed_dim) must be divisible by num_heads
        if d_model % heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by heads ({heads})")

        # Layers for processing the query (EEG)
        self.q_proj = nn.Linear(emb_size, d_model)
        self.fusion_proj = nn.Linear(emb_size, d_model)

        # Layers for processing the key/value (fNIRS)
        # self.k_proj = nn.Linear(11 * 32, d_model) # Assuming fnirs shape after gap is (batch, 11, 32)
        # self.pos = PosEmbedding(input_shape=(None, 11, 32))
        self.k_proj = nn.Linear(emb_size, d_model)

        # Multi-Head Attention layer
        self.dot_product_attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=heads, dropout=drop, batch_first=True)

    def forward(self, inputs):
        """
        Forward pass for the EEG-fNIRS attention mechanism.
        Args:
            inputs (tuple): A tuple containing EEG and fNIRS tensors.
        Returns:
            tuple: A tuple containing fusion_output, fnirs_weighted, attention_weights, and ef_loss.
        """
        eeg, fnirs = inputs
        batch_size = eeg.shape[0]

        # Flatten EEG and project
        q_eeg = eeg.view(batch_size, -1)
        
        # Project for the main fusion output path
        fusion_output = self.fusion_proj(q_eeg)

        # Project for the attention mechanism
        q_eeg_attn = self.q_proj(q_eeg)
        # q_eeg_attn = q_eeg_attn.unsqueeze(1)  # (batch_size, 1, d_model)

        # Reshape fNIRS, add positional encoding, and project
        # k_fnirs = fnirs.view(batch_size, 11, -1) # Reshape to (batch, 11, features)
        # k_fnirs = self.pos(k_fnirs)
        # k_fnirs = self.k_proj(k_fnirs.view(batch_size, -1)).view(batch_size, 11, -1) # Project and reshape back
        k_fnirs = fnirs.view(batch_size, -1)
        k_fnirs = self.k_proj(k_fnirs)

        # Multi-Head Attention
        # PyTorch's MHA returns (output, weights)
        fnirs_weighted, attention_weights = self.dot_product_attention(q_eeg_attn, k_fnirs, k_fnirs)
        
        # Average attention weights across heads and query dimension
        attention_weights = torch.mean(attention_weights, dim=1) # Average over heads

        # Reduce mean for loss calculation
        # q_eeg_loss = torch.mean(q_eeg_attn, dim=1)
        # fnirs_weighted_loss = torch.mean(fnirs_weighted, dim=1)

        # Calculate Pearson correlation loss
        # ef_loss = pearson_r(q_eeg_loss, fnirs_weighted_loss)
        ef_loss = pearson_r(q_eeg_attn, fnirs_weighted)

        return fusion_output, fnirs_weighted.squeeze(1), attention_weights, ef_loss


class FGA(nn.Module):
    """
    Fusion Gate Attention (FGA) layer.
    """
    def __init__(self, in_chan, tem_kernel_size):
        super(FGA, self).__init__()
        # 3D convolution to create an attention map from fNIRS features
        self.channel_pooling = nn.Conv3d(in_channels=in_chan, out_channels=1, kernel_size=(3, 3, tem_kernel_size), stride=1, padding='same')
        # A trainable parameter for the residual connection
        self.residual_para = nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        """
        Forward pass for the FGA mechanism.
        Args:
            inputs (tuple): A tuple containing eeg_fusion, eeg, and fnirs tensors.
        Returns:
            tuple: A tuple containing the fga_feature and the fga_loss.
        """
        eeg_fusion, eeg, fnirs = inputs
        
        # Generate attention map from fNIRS
        fnirs_attention = self.channel_pooling(fnirs)
        # Global Average Pooling over temporal dimension
        fnirs_attention_map = torch.mean(fnirs_attention, dim=-1, keepdim=True)
        # fnirs_attention_map = torch.mean(fnirs_attention_map, dim=1)

        # Normalize attention map
        fnirs_attention_map_norm = torch.sigmoid(fnirs_attention_map)

        # Apply attention to the fusion feature stream
        eeg_fusion_guided = eeg_fusion * fnirs_attention_map_norm

        # Calculate weighted sum for the residual connection
        residual_para_norm = torch.sigmoid(self.residual_para)
        eeg_add = residual_para_norm * eeg + (1 - residual_para_norm) * eeg_fusion

        # Combine the guided fusion stream and the residual stream
        fga_feature = eeg_fusion_guided + eeg_add

        # --- Loss Calculation ---
        # Flatten features for Pearson correlation
        eeg_plcc = torch.mean(eeg, dim=(-1, 1)).view(eeg.shape[0], -1)
        fnirs_attention_map_norm_plcc = fnirs_attention_map_norm.view(fnirs_attention_map_norm.shape[0], -1)

        # Calculate FGA loss
        fga_loss = pearson_r(eeg_plcc, fnirs_attention_map_norm_plcc)

        return fga_feature, fga_loss


class ConvBlock(nn.Module):
    """
    A convolutional block that processes EEG, fNIRS, and a fusion stream,
    and applies Fusion Gate Attention (FGA).
    """
    def __init__(self, eeg_in, eeg_filter, eeg_size, eeg_stride, eeg_padding,
                 fnirs_in, fnirs_filter, fnirs_size, fnirs_stride, fnirs_padding,
                 eegfusion_in, eegfusion_filter, eegfusion_size, eegfusion_stride, eegfusion_padding,
                 tem_kernel_size, padding):
        super(ConvBlock, self).__init__()

        # EEG processing path
        self.eeg_conv = nn.Conv3d(eeg_in, eeg_filter, kernel_size=eeg_size, stride=eeg_stride, padding=eeg_padding)
        self.eeg_bn = nn.BatchNorm3d(eeg_filter)
        self.eeg_act = nn.ELU()

        # fNIRS processing path
        self.fnirs_conv = nn.Conv3d(fnirs_in, fnirs_filter, kernel_size=fnirs_size, stride=fnirs_stride, padding=fnirs_padding)
        self.fnirs_bn = nn.BatchNorm3d(fnirs_filter)
        self.fnirs_act = nn.ELU()

        # Fusion stream processing path
        self.eegfusion_conv = nn.Conv3d(eegfusion_in, eegfusion_filter, kernel_size=eegfusion_size, stride=eegfusion_stride, padding=eegfusion_padding)
        self.eegfusion_bn = nn.BatchNorm3d(eegfusion_filter)
        self.eegfusion_act = nn.ELU()

        # Fusion Gate Attention module
        self.fga = FGA(eeg_filter, tem_kernel_size)

    def forward(self, inputs):
        """
        Forward pass for the convolutional block.
        """
        eegfusion, eeg, fnirs = inputs

        # Process EEG stream
        eeg_feature = self.eeg_act(self.eeg_bn(self.eeg_conv(eeg)))
        
        # Process fNIRS stream
        fnirs_feature = self.fnirs_act(self.fnirs_bn(self.fnirs_conv(fnirs)))
        
        # Process Fusion stream
        eegfusion_feature = self.eegfusion_act(self.eegfusion_bn(self.eegfusion_conv(eegfusion)))

        # Apply FGA
        eegfusion_fga, fga_loss = self.fga((eegfusion_feature, eeg_feature, fnirs_feature))

        return eegfusion_fga, eeg_feature, fnirs_feature, fga_loss
        

class STANet(nn.Module):
    """
    The main Spatio-Temporal Attention Network (STA-Net) model.
    """
    def __init__(self):
        super(STANet, self).__init__()
        
        # First Convolutional Block
        self.conv_block1 = ConvBlock(
            eeg_in=1, eeg_filter=16, eeg_size=(2, 2, 13), eeg_stride=(2, 2, 6), eeg_padding=(0,0,6),
            fnirs_in=2, fnirs_filter=16, fnirs_size=(2, 2, 5), fnirs_stride=(2, 2, 2), fnirs_padding=(0,0,2),
            eegfusion_in=1, eegfusion_filter=16, eegfusion_size=(2, 2, 13), eegfusion_stride=(2, 2, 6), eegfusion_padding=(0,0,6),
            tem_kernel_size=5, padding=0#'same'
        )
        self.dropout1 = nn.Dropout(0.5)

        # Second Convolutional Block
        self.conv_block2 = ConvBlock(
            eeg_in=16, eeg_filter=32, eeg_size=(2, 2, 5), eeg_stride=(2, 2, 2), eeg_padding=(0,0,2),
            fnirs_in=16, fnirs_filter=32, fnirs_size=(2, 2, 3), fnirs_stride=(2, 2, 2), fnirs_padding=(0,0,1),
            eegfusion_in=16, eegfusion_filter=32, eegfusion_size=(2, 2, 5), eegfusion_stride=(2, 2, 2), eegfusion_padding=(0,0,2),
            tem_kernel_size=3, padding=0#'same'
        )
        self.dropout2 = nn.Dropout(0.5)

        # EEG-fNIRS Attention
        self.e_f_attention = EFAttention(emb_size=4*4*32, d_model=256, heads=8, drop=0.5)

        # Dense layers for prediction heads
        self.eegfusion_fc1 = nn.Linear(256, 256)
        self.fnirs_fc1 = nn.Linear(256, 256)
        
        self.eeg_fc1 = nn.Linear(4*4*32, 256) # Flattened size of eeg2 after GAP
        
        self.eegfusion_pred_fc = nn.Linear(256, 1)
        self.fnirs_pred_fc = nn.Linear(256, 1)
        self.eeg_pred_fc = nn.Linear(256, 1)

        # Dense layers for prediction weights
        self.eegfusion_p_weight_fc = nn.Linear(256, 1)
        self.fnirs_p_weight_fc = nn.Linear(256, 1)

    def forward(self, eeg_input, fnirs_input):
        # PyTorch expects (N, C, D, H, W) so we need to adjust dimensions
        # Original EEG: (N, 16, 16, 600, 1) -> (N, 1, 600, 16, 16)
        # Original fNIRS: (N, 11, 16, 16, 30, 2) -> (N, 2, 30, 11, 16, 16) - This is 6D, Conv3D takes 5D
        # Assuming fNIRS input is (N, 16, 16, 30, 2) and we merge the last two dims
        # Or that the 11 is a channel-like dimension. Let's assume the latter for Conv3D.
        # Let's assume fnirs_input is (N, 2, 30, 11, 16, 16) and we process each of the 11 items
        # The original paper's code seems to have a dimension mismatch.
        # For this translation, we assume the input shapes are compatible with Conv3d:
        # eeg_input: (N, 1, 600, 16, 16)
        # fnirs_input: (N, 2, 30, 16, 16) - Assuming the 11 channels are handled differently. 
        # The original TF code has shape (11, 16, 16, 30, 2), which is unusual for a batch.
        # Let's assume the batch is first and the 11 is a dimension to be flattened later.
        # fnirs_input: (N, 11, 16, 16, 30, 2) -> Transpose -> (N, 2, 30, 11, 16, 16)
        # This is complex. For a direct translation, let's assume the TF code's Conv3D on fnirs
        # was a typo and it should have been Conv2D over the 11 channels, or the shape is different.
        # Let's stick to the TF shapes and adjust PyTorch Conv3D inputs.
        # TF Conv3D input: (batch_size, spatial_dim1, spatial_dim2, spatial_dim3, channels)
        # PyTorch Conv3D input: (batch_size, channels, spatial_dim1, spatial_dim2, spatial_dim3)
        # EEG: (N, 1, 16, 16, 600)
        # fNIRS: (N, 2, 11, 16, 16, 30) -> (N, 2*11, 30, 16, 16) -> Let's assume this for now.
        # The most likely interpretation is that the TF code is channel-last and PyTorch is channel-first.
        # EEG: (N, 16, 16, 600, 1) -> permute -> (N, 1, 600, 16, 16)
        # fNIRS: (N, 11, 16, 16, 30, 2) -> permute -> (N, 2, 30, 11, 16, 16) -> This is still 6D.
        # Let's assume the 11 is part of the spatial dimensions for fNIRS.
        # fNIRS: (N, 16, 16, 30, 2) -> permute -> (N, 2, 30, 16, 16)
        
        # Block 1
        eegfusion1, eeg1, fnirs1, fga1_loss = self.conv_block1((eeg_input, eeg_input, fnirs_input))
        eegfusion1, eeg1, fnirs1 = self.dropout1(eegfusion1), self.dropout1(eeg1), self.dropout1(fnirs1)

        # Block 2
        eegfusion2, eeg2, fnirs2, fga2_loss = self.conv_block2((eegfusion1, eeg1, fnirs1))
        
        # Global Average Pooling (GAP) across spatial dimensions
        # eegfusion2 = F.adaptive_avg_pool3d(eegfusion2, (1, 1, 1)).squeeze()
        # eeg2 = F.adaptive_avg_pool3d(eeg2, (1, 1, 1)).squeeze()
        # fnirs2 = F.adaptive_avg_pool3d(fnirs2, (1, 1, 1)).squeeze()
        eegfusion2 = torch.mean(eegfusion2, dim=-1, keepdim=True)
        eeg2 = torch.mean(eeg2, dim=-1, keepdim=True)
        fnirs2 = torch.mean(fnirs2, dim=-1, keepdim=True)

        eegfusion2, eeg2, fnirs2 = self.dropout2(eegfusion2), self.dropout2(eeg2), self.dropout2(fnirs2)

        # EEG-fNIRS Attention
        eegfusion_feature, fnirs_feature, _, ef_loss = self.e_f_attention((eegfusion2, fnirs2))
        
        eegfusion_feature_pweight = F.elu(eegfusion_feature)
        fnirs_feature_pweight = F.elu(fnirs_feature)

        eegfusion_feature_pweight = F.elu(self.eegfusion_fc1(eegfusion_feature_pweight))
        fnirs_feature_pweight = F.elu(self.fnirs_fc1(fnirs_feature_pweight))

        # EEG feature path
        eeg_feature = eeg2.view(eeg2.size(0), -1)
        eeg_feature = F.elu(self.eeg_fc1(eeg_feature))

        # Prediction heads
        eegfusion_pred = self.eegfusion_pred_fc(eegfusion_feature_pweight)
        fnirs_pred = self.fnirs_pred_fc(fnirs_feature_pweight)
        eeg_pred = self.eeg_pred_fc(eeg_feature)

        # Main EEG prediction output with softmax
        # eeg_pred_out = F.softmax(eeg_pred, dim=1)
        eeg_pred_out = F.sigmoid(eeg_pred)

        # Combine fusion and fNIRS predictions for the final output
        # eegfusion_pred = F.softmax(eegfusion_pred, dim=1).unsqueeze(1)
        # fnirs_pred = F.softmax(fnirs_pred, dim=1).unsqueeze(1)
        # eegfusion_pred = F.sigmoid(eegfusion_pred).unsqueeze(1)
        # fnirs_pred = F.sigmoid(fnirs_pred).unsqueeze(1)
        
        the_pred_cat = torch.cat([eegfusion_pred, fnirs_pred], dim=1)

        # Calculate dynamic prediction weights
        eegfusion_p_weight = self.eegfusion_p_weight_fc(eegfusion_feature_pweight)
        fnirs_p_weight = self.fnirs_p_weight_fc(fnirs_feature_pweight)

        p_weight = torch.cat([eegfusion_p_weight, fnirs_p_weight], dim=1)
        p_weight = F.softmax(p_weight, dim=1)#.unsqueeze(-1)

        # Apply weights and sum
        the_pred = the_pred_cat * p_weight
        the_pred_out = torch.mean(the_pred, dim=1)
        the_pred_out = F.sigmoid(the_pred_out)
        # The total loss would be a combination of the main classification loss
        # and the custom correlation losses returned here.
        # Loss = main_loss + (1 - fga1_loss) + (1 - fga2_loss) + (1 - ef_loss)
        custom_losses = (1 - fga1_loss) + (1 - fga2_loss) + (1 - ef_loss)

        return the_pred_out, eeg_pred_out, custom_losses

def make_3d_input_for_stanet(sig, dat_type='emotion'):
    from scipy.interpolate import griddata

    x = np.arange(16)
    y = np.arange(16)
    xx, yy = np.meshgrid(x,y)
    all_points = np.column_stack((xx.ravel(), yy.ravel()))

    if dat_type == 'emotion':
    # eeg (36, 8, 7, 7680)
    # fnirs (36, 8, 26, 371)
    # eeg_input: (N, 1, 600, 16, 16)
    # fnirs_input: (N, 2, 30, 16, 16)
        if sig.shape[2] == 7:
            rg = 7
            known_point_coordinates = np.array([[15,5],[13,10],[12,5],[8,5],[4,5],[3,10],[1,5]])
            out = np.zeros((36,8,1,7680,16,16))
        else:
            rg = 13
            known_point_coordinates = np.array([[15,5],[14,12],[14,2],[12,7],[10,12],[10,2],[8,7],[6,12],[6,2],[4,7],[2,12],[2,2],[1,7]])
            out = np.zeros((36,8,2,371,16,16))
        unknown_point_coordinates = np.array([coord for coord in all_points if coord.tolist() not in known_point_coordinates.tolist()])
        unknown_point_coordinates = unknown_point_coordinates.astype(float)
        for sub in range(36):
            for tri in range(8):
                for t in range(sig.shape[-1]):
                    if rg == 7:
                        img_2d = np.ones((16, 16))
                        known_point_values = sig[sub,tri,:,t]
                        interpol = griddata(points=known_point_coordinates,
                                            values=known_point_values,
                                            xi=unknown_point_coordinates,
                                            fill_value = 0,
                                            method='cubic')
                        
                        for k in range(rg):
                            img_2d[int(known_point_coordinates[k, 0]), int(known_point_coordinates[k, 1])] = known_point_values[k]
                        for k in range(256-rg):
                            img_2d[int(unknown_point_coordinates[k, 0]), int(unknown_point_coordinates[k, 1])] = interpol[k]
                        out[sub,tri,0,t] = img_2d
                    else:
                        for a,b,c in [(0,0,13),(1,13,26)]:
                            img_2d = np.ones((16, 16))
                            known_point_values = sig[sub,tri,b:c,t]
                            interpol = griddata(points=known_point_coordinates,
                                                values=known_point_values,
                                                xi=unknown_point_coordinates,
                                                fill_value = 0,
                                                method='cubic')
                            
                            for k in range(rg):
                                img_2d[int(known_point_coordinates[k, 0]), int(known_point_coordinates[k, 1])] = known_point_values[k]
                            for k in range(256-rg):
                                img_2d[int(unknown_point_coordinates[k, 0]), int(unknown_point_coordinates[k, 1])] = interpol[k]
                            out[sub,tri,a,t] = img_2d
    np.save(f'out{rg}.npy',out)
    return out

if __name__ == '__main__':
    # --- Model Instantiation and Dummy Data Test ---
    
    # Create an instance of the model
    model = STANet()
    
    # NOTE: The input shapes in the original TF code are ambiguous for Conv3D.
    # TF Conv3D input: (batch, d1, d2, d3, channels)
    # PyTorch Conv3D input: (batch, channels, d1, d2, d3)
    # We'll create dummy data assuming the TF->PyTorch permutation.
    
    # Dummy input tensors
    # EEG: TF(16, 16, 600, 1) -> PT(1, 600, 16, 16)
    # fNIRS: TF(11, 16, 16, 30, 2) -> This is the most problematic shape.
    # Let's assume the 11 is a channel-like dimension that gets processed somehow.
    # For this test, we'll create a shape that works with the defined Conv3D layers.
    # fNIRS Input for ConvBlock1: (N, 2, 30, 16, 16)
    batch_size = 4
    dummy_eeg = torch.randn(batch_size, 1, 600, 16, 16)
    dummy_fnirs = torch.randn(batch_size, 2, 30, 16, 16) # Simplified shape

    print(f"Model Architecture:\n{model}")
    
    # Set model to training mode
    model.train()
    
    # Forward pass
    try:
        main_output, eeg_output, custom_loss = model(dummy_eeg, dummy_fnirs)
        
        print("\n--- Forward Pass Successful ---")
        print(f"Main Output Shape: {main_output.shape}")
        print(f"EEG Output Shape: {eeg_output.shape}")
        print(f"Custom Loss Value: {custom_loss.item()}")
        
        # Example of calculating total loss in a training loop
        criterion = nn.CrossEntropyLoss()
        labels = torch.randint(0, 2, (batch_size,))
        
        # Main loss for the combined prediction
        main_loss = criterion(main_output, labels)
        # Separate loss for the EEG-only branch
        eeg_loss = criterion(eeg_output, labels)
        
        # Total loss combines the classification losses and the custom correlation losses
        total_loss = main_loss + eeg_loss + custom_loss
        
        print(f"\nExample Total Loss: {total_loss.item()}")
        
        # To backpropagate:
        # total_loss.backward()
        
    except Exception as e:
        print(f"\n--- An error occurred during the forward pass ---")
        print(e)
        import traceback
        traceback.print_exc()

