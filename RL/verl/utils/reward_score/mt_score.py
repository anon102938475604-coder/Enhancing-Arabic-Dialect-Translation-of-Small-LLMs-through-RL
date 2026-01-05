import re
from typing import Dict, Tuple, Optional, Any
import sacrebleu

# Regex patterns for extracting think content
THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

def think_len_score_dense(len_think: int, L_target: int) -> float:
    """Compute dense saturating reward score for think length.
    
    Args:
        len_think: Length of think content in tokens
        L_target: Current target length from curriculum ladder
        
    Returns:
        Score in [0, 1] range: clip(len_think / L_target, 0, 1)
        Gives partial credit immediately (no dead zone).
    """
    if L_target <= 0:
        return 0.0
    # Dense saturating reward: partial credit for any length
    return max(0.0, min(1.0, len_think / L_target))


def get_curriculum_target(progress: float, 
                         phase1_target: int = 24,
                         phase2_target: int = 48, 
                         phase3_target: int = 96,
                         phase1_end: float = 0.2,
                         phase2_end: float = 0.5) -> int:
    """Compute current target length based on training progress.
    
    Args:
        progress: Training progress ratio (0.0 to 1.0)
        phase1_target: Target tokens for phase 1 (first 20% by default)
        phase2_target: Target tokens for phase 2 (20-50% by default)
        phase3_target: Target tokens for phase 3 (50-100% by default)
        phase1_end: End of phase 1 as progress ratio
        phase2_end: End of phase 2 as progress ratio
        
    Returns:
        Current target length in tokens based on curriculum phase
    """
    if progress < phase1_end:
        return phase1_target
    elif progress < phase2_end:
        return phase2_target
    else:
        return phase3_target


def compute_short_penalty(len_think: int, short_threshold: int = 8) -> float:
    """Compute penalty for very short think traces.
    
    Args:
        len_think: Length of think content in tokens
        short_threshold: Threshold below which penalty is applied
        
    Returns:
        1.0 if len_think < short_threshold, else 0.0
    """
    return 1.0 if len_think < short_threshold else 0.0


def compute_repetition_penalty(think_text: str, n: int = 4) -> float:
    """Compute repetition penalty based on unique n-gram ratio.
    
    Args:
        think_text: Text content from <think> tags
        n: N-gram size (default 4)
        
    Returns:
        Repetition penalty value in [0, 1] range where higher = more repetitive.
        Calculated as 1 - (unique_n_grams / total_n_grams)
    """
    if not think_text or len(think_text) < n:
        return 0.0
    
    # Extract n-grams
    words = think_text.split()
    if len(words) < n:
        return 0.0
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = tuple(words[i:i+n])
        ngrams.append(ngram)
    
    if not ngrams:
        return 0.0
    
    unique_ngrams = len(set(ngrams))
    total_ngrams = len(ngrams)
    
    # Repetition ratio: higher means more repetitive
    rep_ratio = 1.0 - (unique_ngrams / total_ngrams)
    return rep_ratio


def extract_think_content(processed_str: str) -> Optional[str]:
    """Extract content between <think> tags.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Extracted think content, or None if not found
    """
    match = THINK_RE.search(processed_str)
    if match:
        return match.group(1).strip()
    return None


def compute_bleu(lg_pair, ref, pred):
    # Added type check
    pred = pred if isinstance(pred, str) else ""
    
    src_lang = lg_pair.split("-")[0]
    tgt_lang = lg_pair.split("-")[1]
    
    tokenize = "zh" if tgt_lang == "zh" else "ja-mecab" if tgt_lang == "ja" else "13a"
    
    refs = [[ref]]
    sys = [pred]

    bleu_str = str(sacrebleu.corpus_bleu(sys, refs, tokenize=tokenize))  # Note: BLEU tokenize
    bleu_score = re.search(r'BLEU = (\d+\.\d+)', bleu_str).group(1)

    print(f"[BLEU Score] {bleu_score}")
    return float(bleu_score)

def extract_solution(solution_str: str) -> Tuple[str, str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str: # base
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str: # qwen and tower
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    elif "<|start_header_id|>assistant<|end_header_id|>" in solution_str: # llama3
        processed_str = solution_str.split("<|start_header_id|>assistant<|end_header_id|>", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract final answer using XML-style tags
    answer_pattern = r'<translate>(.*?)</translate>'
    matches = list(re.finditer(answer_pattern, processed_str, re.DOTALL))
    
    if not matches:
        print("[Error] No valid answer tags found")
        return None, processed_str
        
    final_answer = matches[-1].group(1).strip()
    return final_answer, processed_str


def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<translate>', 1),
        'answer_end': ('</translate>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed




def compute_score(reward_metric: str,
                 reward_type: str,
                 metric_score: None,
                 lg_pair: None,
                 bleu_threshold: float,
                 comet_threshold: float,
                 solution_str: str, 
                 ground_truth: str,
                 scale_factor: float = 100.0,
                 check_think: bool = True,
                 format_reward: int = 1,
                 answer_reward: float = 1.0,
                 use_think_length_reward: bool = False,
                 think_length_alpha: float = 0.2,
                 training_progress: float = 0.0,
                 think_length_phase1_target: int = 24,
                 think_length_phase2_target: int = 48,
                 think_length_phase3_target: int = 96,
                 think_length_phase1_end: float = 0.2,
                 think_length_phase2_end: float = 0.5,
                 use_short_penalty: bool = True,
                 short_penalty_threshold: int = 8,
                 short_penalty_beta: float = 0.2,
                 use_repetition_penalty: bool = True,
                 repetition_penalty_gamma: float = 0.1,
                 tokenizer: Optional[Any] = None) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: target ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        use_think_length_reward: Whether to enable think-length reward
        think_length_alpha: Weight for length reward (default 0.2)
        training_progress: Training progress ratio (0.0 to 1.0) for curriculum
        think_length_phase1_target: Target tokens for phase 1 (first 20% by default)
        think_length_phase2_target: Target tokens for phase 2 (20-50% by default)
        think_length_phase3_target: Target tokens for phase 3 (50-100% by default)
        think_length_phase1_end: End of phase 1 as progress ratio
        think_length_phase2_end: End of phase 2 as progress ratio
        use_short_penalty: Whether to enable short trace penalty
        short_penalty_threshold: Token threshold below which penalty is applied
        short_penalty_beta: Weight for short penalty (default 0.2)
        use_repetition_penalty: Whether to enable repetition penalty
        repetition_penalty_gamma: Weight for repetition penalty (0.05-0.2 range)
        tokenizer: Tokenizer instance (required when use_think_length_reward=True)
        
    Returns:
        Total score (sum of format, answer, and optional think-length rewards)
    """
    print("\n" + "="*80)
    print(" Processing Training Sample ".center(80, '='))
    

    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Prompt + Response]\n{solution_str}")

    # Validate response structure
    if check_think:
        format_correct = validate_response_structure(processed_str)
        format_score = format_reward if format_correct else -abs(format_reward)
    else:
        format_correct = answer_text != None
        format_score = format_reward if format_correct else -abs(format_reward)

    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")


    # Validate answer content
    answer_score = 0
    if format_correct and answer_text:

        bleu_score = compute_bleu(lg_pair, ground_truth, answer_text)

        if reward_type == 'discrete':
            if reward_metric == 'BLEU':
                if bleu_score > bleu_threshold:
                    answer_score = 2
                else:
                    answer_score = -1.5

            elif reward_metric == 'Model':
                if metric_score is None:
                    raise ValueError("comet_rm is None, enable comet or cometfree use_rm")
                if metric_score > comet_threshold:
                    answer_score = 2
                else:
                    answer_score = -1.5

            elif reward_metric == 'Merge':
                if metric_score is None:
                    raise ValueError("comet_rm is None, enable comet or cometfree use_rm")
                if bleu_score > bleu_threshold and metric_score > comet_threshold:
                    answer_score = 2
                else:
                    answer_score = -1.5

        elif reward_type == 'continuous':
            if reward_metric == 'BLEU':
                answer_score = float(bleu_score) / float(scale_factor)

            elif reward_metric == 'Model':
                if metric_score is None:
                    raise ValueError("comet_rm is None, enable comet or cometfree use_rm")
                answer_score = float(metric_score) / float(scale_factor)


            elif reward_metric == 'Merge':
                if metric_score is None:
                    raise ValueError("comet_rm is None, enable comet or cometfree use_rm")
                answer_score = float(bleu_score) / float(scale_factor) + float(metric_score) / float(scale_factor)

        else:
            raise ValueError("Invalid reward_type, please use discrete or continuous")

        print(f"\n[Content Validation]")
        print(f"Reference: {ground_truth}")
        print(f"Hypothesis: {answer_text}")
        print(f"BLEU Score: {bleu_score}")
        print(f"Metric Model Score: {metric_score}" if metric_score is not None else "") 
        
    else:
        answer_score = -2
        print("\n[Content Validation] Skipped due to format errors or missing answer")

    # Compute think-length reward (only when format is valid)
    think_length_reward = 0.0
    repetition_penalty_value = 0.0
    short_penalty_value = 0.0
    
    if use_think_length_reward and format_correct:
        if tokenizer is None:
            raise ValueError("tokenizer must be provided when use_think_length_reward=True")
        
        think_content = extract_think_content(processed_str)
        if think_content:
            # Compute think length in tokens
            len_think = len(tokenizer.encode(think_content, add_special_tokens=False))
            
            # Get current curriculum target based on training progress
            L_target = get_curriculum_target(
                training_progress,
                phase1_target=think_length_phase1_target,
                phase2_target=think_length_phase2_target,
                phase3_target=think_length_phase3_target,
                phase1_end=think_length_phase1_end,
                phase2_end=think_length_phase2_end
            )
            
            # Compute dense saturating reward
            s_len = think_len_score_dense(len_think, L_target)
            think_length_reward = think_length_alpha * s_len
            
            # Compute short penalty if enabled
            if use_short_penalty:
                short_penalty = compute_short_penalty(len_think, short_penalty_threshold)
                short_penalty_value = short_penalty_beta * short_penalty
            
            # Compute repetition penalty if enabled
            if use_repetition_penalty:
                rep_penalty = compute_repetition_penalty(think_content, n=4)
                repetition_penalty_value = repetition_penalty_gamma * rep_penalty
            
            print(f"\n[Think-Length Reward]")
            print(f"  Training progress: {training_progress:.4f}")
            print(f"  Curriculum target (L_target): {L_target}")
            print(f"  Think length (tokens): {len_think}")
            print(f"  Length score (s_len): {s_len:.4f}")
            print(f"  Length reward (alpha * s_len): {think_length_reward:.4f}")
            if use_short_penalty:
                print(f"  Short penalty: {short_penalty:.4f}")
                print(f"  Short penalty value (beta * penalty): {short_penalty_value:.4f}")
            if use_repetition_penalty:
                print(f"  Repetition penalty: {rep_penalty:.4f}")
                print(f"  Repetition penalty value (gamma * rep): {repetition_penalty_value:.4f}")
        else:
            print(f"\n[Think-Length Reward] Think content not found, skipping")

    total_score = format_score + answer_score + think_length_reward - repetition_penalty_value - short_penalty_value
    print("\n" + "-"*80)
    print(f" Reward Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    if use_think_length_reward:
        print(f"  Think-Length: {think_length_reward:.4f}")
        if use_short_penalty:
            print(f"  Short Penalty: -{short_penalty_value:.4f}")
        if use_repetition_penalty:
            print(f"  Repetition Penalty: -{repetition_penalty_value:.4f}")
    print(f"  Total: {total_score:.4f}")
    print("="*80 + "\n")

    return total_score



def compute_score_val_bleu(solution_str: str, 
                 ground_truth: str,
                 lg_pair:str, 
                 format_reward: int = 1,
                 answer_reward: float = 1.0) :
    """Computes comprehensive score for model response.
    
    Args:
        solution_str: Raw model response string
        ground_truth: target ground truth data
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "="*80)
    print(" Processing Test Sample ".center(80, '='))
    
    solution_text = ground_truth
    # Extract model answer
    answer_text, processed_str = extract_solution(solution_str)
    print(f"\n[Prompt + Response]\n{solution_str}")

    if answer_text:
        pred_status = compute_bleu(lg_pair, ground_truth, answer_text)
        print(f"Reference: {solution_text}")
        print(f"Hypothesis: {answer_text}")
    else:
        pred_status = compute_bleu(lg_pair, ground_truth, answer_text)
        print(f"Reference: {solution_text}")
        print(f"Hypothesis: {processed_str}")
        

    total_score = pred_status
    print("\n" + "-"*80)
    print(f"BLEU Score: {total_score}")


    return total_score