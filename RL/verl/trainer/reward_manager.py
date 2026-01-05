from verl.utils.reward_score import mt_score
from verl import DataProto
import torch
def _select_rm_score_fn(data_source):

    return mt_score.compute_score


def _select_metric_score_fn(data_source):

    return mt_score.compute_score_val_bleu


class RewardManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine, config) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.reward_type = config.algorithm.reward_type
        self.reward_metric = config.algorithm.reward_metric
        assert self.reward_type in ['discrete', 'continuous'], "reward_type must be discrete or continue"
        assert self.reward_metric in ['BLEU', 'Model', 'Merge'], "reward_metric must be BLEU or Model or Merge" 
        self.bleu_threshold = config.algorithm.bleu_threshold 
        self.comet_threshold = config.algorithm.comet_threshold
        self.scale_factor = config.algorithm.reward_continuous_scale
        self.check_think = config.algorithm.check_think
        
        # Training step tracking for curriculum
        self.current_step = 0
        # Try to get total_training_steps from config, compute if not available
        if hasattr(config.trainer, 'total_training_steps') and config.trainer.total_training_steps is not None:
            self.total_training_steps = config.trainer.total_training_steps
        elif hasattr(config.trainer, 'total_epochs') and hasattr(config, 'data'):
            # Will be updated when trainer initializes, use a placeholder for now
            self.total_training_steps = 1  # Placeholder, will be updated
        else:
            self.total_training_steps = 1  # Placeholder
        
        # Think-length reward parameters (with defaults for backward compatibility)
        self.use_think_length_reward = getattr(config.algorithm, 'use_think_length_reward', False)
        self.think_length_alpha = getattr(config.algorithm, 'think_length_alpha', 0.2)
        # Curriculum ladder parameters
        self.think_length_phase1_target = getattr(config.algorithm, 'think_length_phase1_target', 24)
        self.think_length_phase2_target = getattr(config.algorithm, 'think_length_phase2_target', 48)
        self.think_length_phase3_target = getattr(config.algorithm, 'think_length_phase3_target', 96)
        self.think_length_phase1_end = getattr(config.algorithm, 'think_length_phase1_end', 0.2)
        self.think_length_phase2_end = getattr(config.algorithm, 'think_length_phase2_end', 0.5)
        # Short penalty parameters
        self.use_short_penalty = getattr(config.algorithm, 'use_short_penalty', True)
        self.short_penalty_threshold = getattr(config.algorithm, 'short_penalty_threshold', 8)
        self.short_penalty_beta = getattr(config.algorithm, 'short_penalty_beta', 0.2)
        # Repetition penalty parameters
        self.use_repetition_penalty = getattr(config.algorithm, 'use_repetition_penalty', True)
        self.repetition_penalty_gamma = getattr(config.algorithm, 'repetition_penalty_gamma', 0.1)
    
    def update_step(self, step: int) -> None:
        """Update current training step for curriculum computation.
        
        Args:
            step: Current global training step
        """
        self.current_step = step
    
    def update_total_steps(self, total_steps: int) -> None:
        """Update total training steps for curriculum computation.
        
        Args:
            total_steps: Total number of training steps
        """
        self.total_training_steps = max(1, total_steps)  # Ensure at least 1 to avoid division by zero

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_rm_score_fn(data_source)
            if 'comet_rm' in data_item.batch.keys():
                metric_score = float(data_item.batch['comet_rm'])
            elif 'comet_free_rm' in data_item.batch.keys():
                metric_score = float(data_item.batch['comet_free_rm'])
            else:
                metric_score = None
                print("No model-based metric score found, use BLEU")
            lg_pair = data_item.non_tensor_batch['lg']

            # Compute training progress for curriculum
            training_progress = min(1.0, max(0.0, self.current_step / max(1, self.total_training_steps)))
            
            score = compute_score_fn(reward_type = self.reward_type, reward_metric = self.reward_metric, \
                metric_score = metric_score, lg_pair=lg_pair, bleu_threshold = self.bleu_threshold, comet_threshold = self.comet_threshold, \
                 solution_str=sequences_str, ground_truth=ground_truth, scale_factor = self.scale_factor, check_think = self.check_think, \
                 use_think_length_reward = self.use_think_length_reward, think_length_alpha = self.think_length_alpha, \
                 training_progress = training_progress, \
                 think_length_phase1_target = self.think_length_phase1_target, \
                 think_length_phase2_target = self.think_length_phase2_target, \
                 think_length_phase3_target = self.think_length_phase3_target, \
                 think_length_phase1_end = self.think_length_phase1_end, \
                 think_length_phase2_end = self.think_length_phase2_end, \
                 use_short_penalty = self.use_short_penalty, \
                 short_penalty_threshold = self.short_penalty_threshold, \
                 short_penalty_beta = self.short_penalty_beta, \
                 use_repetition_penalty = self.use_repetition_penalty, repetition_penalty_gamma = self.repetition_penalty_gamma, \
                 tokenizer = self.tokenizer)

            reward_tensor[i, valid_response_length - 1] = score

        return reward_tensor


class ValidManager():
    """The reward manager.
    """

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch['prompts']

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            sequences_str = self.tokenizer.decode(sequences)
            
            

            ground_truth = data_item.non_tensor_batch['reward_model']['ground_truth']

            # select rm_score
            data_source = data_item.non_tensor_batch['data_source']
            compute_score_fn = _select_metric_score_fn(data_source)

            lg_pair = data_item.non_tensor_batch['lg']
            score = compute_score_fn(solution_str=sequences_str, ground_truth=ground_truth,lg_pair=lg_pair)
            reward_tensor[i, valid_response_length - 1] = score

            if "valid_comet_metric" in data_item.batch.keys():
                print("valid_comet_metric: ", float(data_item.batch['valid_comet_metric']))
            if "valid_comet_free_metric" in data_item.batch.keys():
                print("valid_comet_free_metric: ", float(data_item.batch['valid_comet_free_metric']))
            print("="*80 + "\n")


        return reward_tensor
