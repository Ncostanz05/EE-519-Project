from dataclasses import dataclass, field


@dataclass
class PipelineEConfig:
    # Backbone
    backbone_name: str = "microsoft/wavlm-large"
    freeze_backbone: bool = True
    use_lora: bool = False
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Architecture
    embedding_dim: int = 1024          # WavLM-large hidden size
    hidden_dim: int = 256
    age_head_type: str = "ldl"         # "ldl" | "ordinal" | "softmax"
    use_uncertainty_head: bool = False
    mc_dropout_samples: int = 10
    use_gender_moe: bool = True

    # Training
    batch_size: int = 16
    num_epochs: int = 30
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    warmup_steps: int = 200
    patience: int = 5
    grad_clip: float = 1.0

    # Loss weights
    age_loss_weight: float = 1.0
    gender_loss_weight: float = 0.3
    ldl_aux_ce_weight: float = 0.2

    # Label smoothing
    adjacency_smoothing_alpha: float = 0.1

    # Augmentation
    use_augmentation: bool = True
    use_speed_perturb: bool = True
    speed_range: tuple = (0.9, 1.1)
    use_gaussian_noise: bool = True
    noise_std_range: tuple = (0.0, 0.005)
    use_random_gain: bool = True
    gain_db_range: tuple = (-6.0, 6.0)
    use_time_masking: bool = True
    time_mask_max_frames: int = 3000   # ~0.19s at 16kHz
    num_time_masks: int = 2

    # Data
    cv_csv: str = "commonvoice-v24_en-AU/commonvoice-v24_en-AU.csv"
    cv_audio_dir: str = "commonvoice-v24_en-AU/audio_files"
    train_frac: float = 0.70
    val_frac: float = 0.15
    seed: int = 42

    # Calibration
    calibrate: bool = True

    # Output
    output_dir: str = "pipeline_e_checkpoints"

    def num_age_outputs(self) -> int:
        return 6 if self.age_head_type == "ordinal" else 7
