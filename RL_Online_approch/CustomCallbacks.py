import numpy as np
import torch
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3 import DQN, A2C, PPO

# State constants (matching v0_turbine_env.py)
BASELINE  = 0
DEVIATION = 1
PLATEAU   = 2

class TurbineEvalCallback(EvalCallback):
    """
    Custom evaluation callback that tracks Q-values separately for different turbine states.
    Logs average Q-values for BASELINE and DEVIATION/PLATEAU states to TensorBoard.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def _on_step(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._evaluate_with_q_values()
            continue_training = super()._on_step()
        
        return continue_training
    
    def _evaluate_with_q_values(self):
        """
        Run evaluation episodes while tracking Q-values per state type.
        """
        # Track Q-values per state type
        values_baseline = [] # BASELINE
        q_values_baseline_0 = []
        q_values_baseline_1 = []
        values_alert = []  # DEVIATION or PLATEAU
        q_values_alert_0 = []
        q_values_alert_1 = []
        
        for _ in range(self.n_eval_episodes):
            obs = self.eval_env.reset()[0]
            done = False
            
            while not done:
                # Get action and value
                action, value, q_value_0, q_value_1 = self._predict_with_q_value(obs)
                
                # Get current state from environment
                # Access the unwrapped environment to get the state
                try:
                    env_state = self.eval_env.envs[0].unwrapped.state
                except (AttributeError, IndexError):
                    try:
                        env_state = self.eval_env.unwrapped.state
                    except AttributeError:
                        env_state = None
                
                # Store Q-value based on state type
                if env_state is not None and value is not None:
                    if env_state == BASELINE:
                        values_baseline.append(value)
                        if q_value_0 is not None and q_value_1 is not None:
                            q_values_baseline_0.append(q_value_0)
                            q_values_baseline_1.append(q_value_1)
                    elif env_state == DEVIATION or env_state == PLATEAU:
                        values_alert.append(value)
                        if q_value_0 is not None and q_value_1 is not None:
                            q_values_alert_0.append(q_value_0)
                            q_values_alert_1.append(q_value_1)
                
                # Take step in environment (VecEnv expects list of actions)
                step_result = self.eval_env.step([action])
                
                # Handle both old gym (4 values) and new gymnasium (5 values) API
                if len(step_result) == 5:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated[0] or truncated[0]
                else:
                    obs, reward, done, info = step_result
                    done = done[0]
                
                obs = obs[0]
                
        # Compute average Q-values per state type
        avg_value_baseline = np.mean(values_baseline) if values_baseline else 0.0
        avg_q_baseline_0 = np.mean(q_values_baseline_0) if q_values_baseline_0 else None
        avg_q_baseline_1 = np.mean(q_values_baseline_1) if q_values_baseline_1 else None
        avg_value_alert = np.mean(values_alert) if values_alert else 0.0
        avg_q_alert_0 = np.mean(q_values_alert_0) if q_values_alert_0 else None
        avg_q_alert_1 = np.mean(q_values_alert_1) if q_values_alert_1 else None
        
        # Log to TensorBoard
        if self.logger is not None:
            self.logger.record("eval/avg_value_baseline", avg_value_baseline)
            self.logger.record("eval/avg_value_alert", avg_value_alert)
            self.logger.record("eval/n_baseline_steps", len(values_baseline))
            self.logger.record("eval/n_alert_steps", len(values_alert))
            # DQN-specific Q-value logging
            if avg_q_baseline_0 is not None and avg_q_baseline_1 is not None:
                self.logger.record("eval/avg_q_baseline_0", avg_q_baseline_0)
                self.logger.record("eval/avg_q_baseline_1", avg_q_baseline_1)
            if avg_q_alert_0 is not None and avg_q_alert_1 is not None:
                self.logger.record("eval/avg_q_alert_0", avg_q_alert_0)
                self.logger.record("eval/avg_q_alert_1", avg_q_alert_1)
        
        if self.verbose > 0:
            print(f"Eval epoch {self.n_calls // self.eval_freq}:")
            print(f"  Avg Value (Baseline): {avg_value_baseline:.4f} ({len(values_baseline)} steps)")
            print(f"  Avg Value (Alert): {avg_value_alert:.4f} ({len(values_alert)} steps)")
    
    def _predict_with_q_value(self, obs):
        """
        Predict action and extract Q-value for that action.
        Returns (action, q_value, q_value_0, q_value_1) for DQN,
        or (action, value) for A2C/PPO.
        """
        model = self.model
        
        # Convert observation to tensor (similar to the MonitorQValueCallback example)
        obs_array = np.array(obs)
        # Flatten the observation if it's multi-dimensional (e.g., from frame stacking)
        if obs_array.ndim > 1:
            obs_array = obs_array.flatten()
        
        obs_tensor = torch.tensor(obs_array, device=model.device).float()
        # Ensure batch dimension
        if obs_tensor.dim() == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        
        with torch.no_grad():
            if isinstance(model, DQN):
                # For DQN, get Q-values from the q-network
                q_values = model.q_net(obs_tensor)
                greedy_action = q_values.argmax(dim=1).item()
                q_value = q_values[0, greedy_action].item()
                q_value_0 = q_values[0, 0].item()
                q_value_1 = q_values[0, 1].item()
                return greedy_action, q_value, q_value_0, q_value_1
            
            elif isinstance(model, (A2C, PPO)):
                # For A2C/PPO, use the value function as a proxy
                # Get action from policy
                action, _ = model.predict(obs, deterministic=self.deterministic)
                
                # Get value estimate
                features = model.policy.extract_features(obs_tensor)
                if hasattr(model.policy, 'mlp_extractor'):
                    latent_vf = model.policy.mlp_extractor.forward_critic(features)
                    values = model.policy.value_net(latent_vf)
                    value = values[0].item()
                else:
                    value = None
                
                return action, value, None, None
            
            else:
                # Fallback for other model types
                action, _ = model.predict(obs, deterministic=self.deterministic)
                return action, None, None, None
