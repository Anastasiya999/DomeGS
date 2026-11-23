#
# Optimization hyperparameters for sky reconstruction.
# Structured like arguments_games, but lives with gs_sky (not GaMeS).
#

from arguments import ParamGroup



class OptimizationParamsSky(ParamGroup):
    """Vanilla 3DGS optimization knobs plus sky-specific extras."""
 
    def __init__(self, parser):
        # --- schedule ---
        self.iterations = 30_000
 
        # --- learning rates (read by training_setup / update_learning_rate) ---
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
 
        # --- photometric loss ---
        self.lambda_dssim = 0.2
        self.random_background = False
 
        # --- densification ---
        self.percent_dense = 0.01
        self.densification_interval = 100   # bear run passed --densification_interval 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        self.max_points = 0
 
        # --- opacity reset (read by the Gaussian model) ---
        self.opacity_reset_cap = 0.01
        self.opacity_reset_floor = 0.006
        self.bg_grad_protect_threshold = 1e-5
 
        # --- seed-shell regularization ---
        # The bear run never defined these, so the loop's getattr fallbacks
        # (1.0 / 0.1 / 30000) were in effect. Made explicit here.
        self.lambda_seed_shell = 1.0
        self.lambda_seed_planarity = 0.1
        self.seed_shell_loss_until_iter = 30_000
 
        super().__init__(parser, "Optimization Parameters")
