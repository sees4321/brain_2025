import numpy as np
import random
import torch
import torch.nn.functional as F

from torch.fft import fftn
from torch.nn.modules.loss import _Loss
from torch.utils.data import Dataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def ManualSeed(seed:int,deterministic=False):
    # random seed 고정
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic: # True면 cudnn seed 고정 (정확한 재현 필요한거 아니면 제외)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def split_subjects(test_subject, num_subjects=32, train_size=28):
    subjects = [i for i in range(num_subjects) if i != test_subject]
    random.shuffle(subjects)

    train_subjects = [False]*(num_subjects)
    val_subjects = [False]*(num_subjects)
    for i, v in enumerate(subjects):
        if i < train_size:
            train_subjects[v] = True
        else:
            val_subjects[v] = True

    return train_subjects, val_subjects

def expand_dim_(data:np.ndarray):
    return np.expand_dims(data,2)

class CustomDataSet(Dataset):
    # x_tensor: data
    # y_tensor: label
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.y)

class BimodalDataSet(Dataset):
    # x_tensor: eeg
    # y_tensor: fnirs
    # z_tensor: label
    def __init__(self, x_tensor, y_tensor, z_tensor):
        self.x = x_tensor
        self.y = y_tensor
        self.z = z_tensor
        assert self.x.size(0) == self.y.size(0)
        assert self.x.size(0) == self.z.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]
    
    def __len__(self):
        return len(self.y)

class JukeboxLoss(_Loss):
    """
    Calculate spectral component based on the magnitude of Fast Fourier Transform (FFT).

    Based on:
        Dhariwal, et al. 'Jukebox: A generative model for music.'https://arxiv.org/abs/2005.00341

    Args:
        spatial_dims: number of spatial dimensions.
        fft_signal_size: signal size in the transformed dimensions. See torch.fft.fftn() for more information.
        fft_norm: {``"forward"``, ``"backward"``, ``"ortho"``} Specifies the normalization mode in the fft. See
            torch.fft.fftn() for more information.

        reduction: {``"none"``, ``"mean"``, ``"sum"``}
            Specifies the reduction to apply to the output. Defaults to ``"mean"``.

            - ``"none"``: no reduction will be applied.
            - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
            - ``"sum"``: the output will be summed.
    """

    def __init__(
        self,
        spatial_dims: int,
        fft_signal_size = None, #:tuple[int] | None = None,
        fft_norm: str = "ortho",
    ) -> None:
        super().__init__()

        self.spatial_dims = spatial_dims
        self.fft_signal_size = fft_signal_size
        self.fft_dim = tuple(range(1, spatial_dims + 2))
        self.fft_norm = fft_norm

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input_amplitude = self._get_fft_amplitude(target)
        target_amplitude = self._get_fft_amplitude(input)

        # Compute distance between amplitude of frequency components
        # See Section 3.3 from https://arxiv.org/abs/2005.00341
        loss = F.mse_loss(target_amplitude, input_amplitude, reduction="none")
        loss = loss.sum()
        return loss

    def _get_fft_amplitude(self, images: torch.Tensor) -> torch.Tensor:
        """
        Calculate the amplitude of the fourier transformations representation of the images

        Args:
            images: Images that are to undergo fftn

        Returns:
            fourier transformation amplitude
        """
        img_fft = fftn(images, s=self.fft_signal_size, dim=self.fft_dim, norm=self.fft_norm)

        amplitude = torch.sqrt(torch.real(img_fft) ** 2 + torch.imag(img_fft) ** 2)

        return amplitude

class EarlyStopping:
    r"""
    Args:
        patience (int): loss or score가 개선된 후 기다리는 기간. default: 3
        delta  (float): 개선시 인정되는 최소 변화 수치. default: 0.0
        mode     (str): 개선시 최소/최대값 기준 선정('min' or 'max'). default: 'min'.
        verbose (bool): 메시지 출력. default: True
    """
    def __init__(self, model, patience=3, delta=0.0, mode='min', verbose=False):

        self.early_stop = False
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        
        self.best_score = np.Inf if mode == 'min' else 0
        self.mode = mode
        self.delta = delta
        self.model = model
        self.epoch = 0

    def __call__(self, score, epoch):

        if self.best_score is None:
            self.best_score = score
            self.counter = 0
        elif self.mode == 'min':
            if score < (self.best_score - self.delta):
                self.counter = 0
                self.best_score = score
                # 모델 저장
                torch.save(self.model.state_dict(), f'best_model.pth')
                self.epoch = epoch
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f} & Model saved')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
        elif self.mode == 'max':
            if score > (self.best_score + self.delta):
                self.counter = 0
                self.best_score = score
                # 모델 저장
                torch.save(self.model.state_dict(), f'best_model.pth')
                self.epoch = epoch
                if self.verbose:
                    print(f'[EarlyStopping] (Update) Best Score: {self.best_score:.5f} & Model saved')
            else:
                self.counter += 1
                if self.verbose:
                    print(f'[EarlyStopping] (Patience) {self.counter}/{self.patience}, ' \
                          f'Best: {self.best_score:.5f}' \
                          f', Current: {score:.5f}, Delta: {np.abs(self.best_score - score):.5f}')
                
            
        if self.counter >= self.patience:
            if self.verbose:
                print(f'[EarlyStop Triggered] Best Score: {self.best_score:.5f}')
            # Early Stop
            self.early_stop = True
        else:
            # Continue
            self.early_stop = False

def plot_confusion_matrix(cf:np.ndarray, cls_names:list):
    import matplotlib.pyplot as plt
    import seaborn as sns
    cf = cf[::-1,::-1]
    n_classes = len(cls_names)
    cf_percent = cf.astype('float') / cf.sum(axis=1)[:, np.newaxis]
    labels = (np.asarray(["{0:d}\n({1:.2%})".format(value, P_value)
                      for value, P_value in zip(cf.flatten(),
                                                cf_percent.flatten())])
          ).reshape(n_classes, n_classes)
    plt.figure(figsize=(n_classes+4.5, n_classes+3))
    sns.set(font_scale=1.2)
    heatmap = sns.heatmap(cf, annot=labels, fmt='', cmap='Blues', cbar=True, #YlGnBu
                          annot_kws={"size": 14})

    heatmap.set_xlabel('Predicted Label', fontsize=14, labelpad=10)
    heatmap.set_ylabel('True Label', fontsize=14, labelpad=10)
    heatmap.set_title('Confusion Matrix', fontsize=16, pad=20)

    heatmap.set_xticklabels(cls_names)
    heatmap.set_yticklabels(cls_names, rotation=0)

    plt.tight_layout()
    plt.savefig('multi_class_confusion_matrix.png', dpi=300)
    plt.show()