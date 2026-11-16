import torch
import torch.nn.functional as F

def calc_q2n(I_GT, I_F, Q_blocks_size, Q_shift):
    """
    Q2n index calculation for image fusion quality assessment.
    
    Args:
        I_GT: Ground-Truth image (torch.Tensor), shape [C, H, W]
        I_F: Fused Image (torch.Tensor), shape [C, H, W]
        Q_blocks_size: Block size for local Q-index calculation
        Q_shift: Block shift for local Q-index calculation
        
    Returns:
        Q2n_index: Overall Q2n index value
        Q2n_index_map: Map of Q2n values, shape [stepx, stepy]
    """
    # Ensure input tensors are float and in CHW format
    I_GT = I_GT.float()
    I_F = I_F.float()
    
    C, N1, N2 = I_GT.shape
    size2 = Q_blocks_size
    
    stepx = (N1 + Q_shift - 1) // Q_shift  # ceil division
    stepy = (N2 + Q_shift - 1) // Q_shift
    
    if stepy <= 0:
        stepy = 1
        stepx = 1
    
    est1 = (stepx - 1) * Q_shift + Q_blocks_size - N1
    est2 = (stepy - 1) * Q_shift + Q_blocks_size - N2
    
    # Pad images if needed
    if est1 > 0 or est2 > 0:
        # Pad with reflection
        pad_width = (est2 // 2, est2 - est2 // 2, est1 // 2, est1 - est1 // 2)
        I_GT = F.pad(I_GT, pad_width, mode='reflect')
        I_F = F.pad(I_F, pad_width, mode='reflect')
        
    C, N1, N2 = I_GT.shape
    
    # Ensure number of channels is power of 2
    if (torch.log2(torch.tensor(C)) % 1) != 0:
        Ndif = 2**torch.ceil(torch.log2(torch.tensor(C))).int() - C
        if Ndif > 0:
            pad_channels = torch.zeros(Ndif, N1, N2, device=I_GT.device, dtype=I_GT.dtype)
            I_GT = torch.cat([I_GT, pad_channels], dim=0)
            I_F = torch.cat([I_F, pad_channels], dim=0)
    
    C, N1, N2 = I_GT.shape
    
    # Initialize Q2n map
    valori = torch.zeros(stepx, stepy, C, device=I_GT.device, dtype=torch.float32)
    
    # Process each block
    for j in range(stepx):
        for i in range(stepy):
            # Calculate block coordinates
            x_start = j * Q_shift
            x_end = x_start + Q_blocks_size
            y_start = i * Q_shift
            y_end = y_start + size2
            
            # Extract blocks
            block_GT = I_GT[:, x_start:x_end, y_start:y_end]
            block_F = I_F[:, x_start:x_end, y_start:y_end]
            
            # Calculate onion quality for this block
            o = onions_quality(block_GT, block_F, Q_blocks_size)
            valori[j, i, :] = o
    
    # Calculate Q2n map and index
    Q2n_index_map = torch.sqrt(torch.sum(valori**2, dim=2))
    Q2n_index = torch.mean(Q2n_index_map)
    
    return Q2n_index, Q2n_index_map


def onions_quality(dat1, dat2, size1):
    """
    Auxiliary function for onions quality calculation.
    
    Args:
        dat1: Ground truth block (C, H, W)
        dat2: Fused image block (C, H, W)
        size1: Block size
        
    Returns:
        q: Quality score for the block
    """
    # Modify dat2: first channel remains, others are negated
    dat2_mod = dat2.clone()
    if dat2_mod.shape[0] > 1:
        dat2_mod[1:, :, :] = -dat2_mod[1:, :, :]
    
    C, H, W = dat1.shape
    
    # Block normalization
    dat1_norm = []
    dat2_norm = []
    
    for i in range(C):
        a1 = dat1[i:i+1, :, :]
        a1_norm, s, t = norm_blocco(a1)
        dat1_norm.append(a1_norm)
        
        # Normalize dat2
        a2 = dat2_mod[i:i+1, :, :]
        if s == 0:
            if i == 0:
                a2_norm = a2 - s + 1
            else:
                a2_norm = -(-a2 - s + 1)
        else:
            if i == 0:
                a2_norm = ((a2 - s) / t) + 1
            else:
                a2_norm = -(((-a2 - s) / t) + 1)
        
        dat2_norm.append(a2_norm)
    
    dat1 = torch.cat(dat1_norm, dim=0)
    dat2 = torch.cat(dat2_norm, dim=0)
    
    # Calculate means
    m1 = torch.mean(dat1, dim=(1, 2))
    m2 = torch.mean(dat2, dim=(1, 2))
    
    # Calculate moduli
    mod_q1m = torch.sqrt(torch.sum(m1**2))
    mod_q2m = torch.sqrt(torch.sum(m2**2))
    
    mod_q1 = torch.sqrt(torch.sum(dat1**2, dim=0))
    mod_q2 = torch.sqrt(torch.sum(dat2**2, dim=0))
    
    # Calculate terms
    termine2 = mod_q1m * mod_q2m
    termine4 = mod_q1m**2 + mod_q2m**2
    
    int1 = (H*W) / ((H*W) - 1) * torch.mean(mod_q1**2)
    int2 = (H*W) / ((H*W) - 1) * torch.mean(mod_q2**2)
    termine3 = int1 + int2 - (H*W) / ((H*W) - 1) * (mod_q1m**2 + mod_q2m**2)
    
    mean_bias = 2 * termine2 / termine4
    
    if torch.isclose(termine3, torch.tensor(0.0)):
        q = torch.zeros(C, device=dat1.device)
        q[-1] = mean_bias
    else:
        cbm = 2 / termine3
        
        # Onion multiplication
        qu = onion_mult2D(dat1, dat2)
        qm = onion_mult(m1, m2)
        
        qv = torch.zeros(C, device=dat1.device)
        for i in range(C):
            qv[i] = (H*W) / ((H*W) - 1) * torch.mean(qu[i, :, :])
        
        q = qv - (H*W) / ((H*W) - 1) * qm
        q = q * mean_bias * cbm
    
    return q


def norm_blocco(block):
    """
    Normalize a single channel block.
    
    Args:
        block: Single channel block (1, H, W)
        
    Returns:
        norm_block: Normalized block
        s: Mean value
        t: Standard deviation
    """
    s = torch.mean(block)
    t = torch.std(block, unbiased=False)
    
    if t == 0:
        norm_block = block - s + 1
    else:
        norm_block = (block - s) / t + 1
    
    return norm_block, s, t


def onion_mult2D(onion1, onion2):
    """
    2D onion multiplication for multi-channel tensors.
    
    Args:
        onion1: First tensor (C, H, W)
        onion2: Second tensor (C, H, W)
        
    Returns:
        ris: Result of onion multiplication
    """
    C, H, W = onion1.shape
    
    if C == 1:
        return onion1 * onion2
    
    L = C // 2
    a = onion1[:L, :, :]
    b = onion1[L:, :, :]
    b = torch.cat([b[:1, :, :], -b[1:, :, :]], dim=0) if L > 1 else b
    
    c = onion2[:L, :, :]
    d = onion2[L:, :, :]
    d = torch.cat([d[:1, :, :], -d[1:, :, :]], dim=0) if L > 1 else d
    
    if C == 2:
        ris = torch.cat([a*c - d*b, a*d + c*b], dim=0)
    else:
        ris1 = onion_mult2D(a, c)
        ris2 = onion_mult2D(d, torch.cat([b[:1, :, :], -b[1:, :, :]], dim=0) if L > 1 else b)
        ris3 = onion_mult2D(torch.cat([a[:1, :, :], -a[1:, :, :]], dim=0) if L > 1 else a, d)
        ris4 = onion_mult2D(c, b)
        
        aux1 = ris1 - ris2
        aux2 = ris3 + ris4
        ris = torch.cat([aux1, aux2], dim=0)
    
    return ris


def onion_mult(onion1, onion2):
    """
    Onion multiplication for vectors.
    
    Args:
        onion1: First vector (C,)
        onion2: Second vector (C,)
        
    Returns:
        ris: Result of onion multiplication
    """
    N = len(onion1)
    
    if N == 1:
        return onion1 * onion2
    
    L = N // 2
    a = onion1[:L]
    b = onion1[L:]
    b = torch.cat([b[:1], -b[1:]]) if L > 1 else b
    
    c = onion2[:L]
    d = onion2[L:]
    d = torch.cat([d[:1], -d[1:]]) if L > 1 else d
    
    if N == 2:
        ris = torch.cat([a*c - d*b, a*d + c*b])
    else:
        ris1 = onion_mult(a, c)
        ris2 = onion_mult(d, torch.cat([b[:1], -b[1:]]) if L > 1 else b)
        ris3 = onion_mult(torch.cat([a[:1], -a[1:]]) if L > 1 else a, d)
        ris4 = onion_mult(c, b)
        
        aux1 = ris1 - ris2
        aux2 = ris3 + ris4
        ris = torch.cat([aux1, aux2])
    
    return ris


# 使用示例
if __name__ == "__main__":
    # 创建测试数据
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 输入应为 [C, H, W] 格式
    gt_image = torch.rand(16, 2000, 2000, device=device)
    fused_image = torch.rand(16, 2000, 2000, device=device)
    
    # 计算Q2n指标
    q2n_index, q2n_map = calc_q2n(gt_image, gt_image, Q_blocks_size=512, Q_shift=512)
    
    print(f"Q2n Index: {q2n_index.item():.4f}")
    print(f"Q2n Map Shape: {q2n_map.shape}")