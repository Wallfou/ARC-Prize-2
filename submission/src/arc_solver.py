from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter
from arc_mask import completion_labels

import arc_config

import gc
import os
import io
import time
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling

import logging
from contextlib import redirect_stdout, redirect_stderr

from peft import get_peft_model_state_dict, set_peft_model_state_dict

import bz2
import pickle

logging.disable(logging.WARNING)

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15
IM_START_ID = 14
NEWLINE_ID = 10


def init_vocab(tokenizer, rank=0):
    """Resolve the special token ids from the tokenizer actually in use.

    NVARC hardcoded these for the 4B checkpoint's 16-token vocabulary. The 2B
    checkpoint is cut from a different base model, so the ids need not match --
    and if `user`/`assistant` resolve elsewhere, the completion-only collator
    finds no turn boundaries, masks every label, and the loss becomes NaN over
    zero targets. Derive them instead, and verify against a real encoding.
    """
    global ARC_TOKENS, USER_TOKEN_ID, ASSISTANT_TOKEN_ID, PAD_ID, EOS_ID
    global IM_START_ID, NEWLINE_ID

    def tid(tok, default=None):
        try:
            i = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            return default
        return default if i is None or i < 0 else int(i)

    USER_TOKEN_ID = tid("user", USER_TOKEN_ID)
    ASSISTANT_TOKEN_ID = tid("assistant", ASSISTANT_TOKEN_ID)
    EOS_ID = tid("<|im_end|>", EOS_ID)
    IM_START_ID = tid("<|im_start|>", IM_START_ID)
    NEWLINE_ID = tid("Ċ", NEWLINE_ID)
    PAD_ID = (int(tokenizer.pad_token_id)
              if getattr(tokenizer, "pad_token_id", None) is not None
              else tid("<|endoftext|>", PAD_ID))

    digits = [tid(str(d)) for d in range(10)]
    newline = tid("Ċ")
    ARC_TOKENS = [t for t in digits + [newline, EOS_ID] if t is not None]

    print(f"[Rank {rank}] vocab: user={USER_TOKEN_ID} assistant={ASSISTANT_TOKEN_ID} "
          f"eos={EOS_ID} pad={PAD_ID} arc_tokens={ARC_TOKENS}")

    # Verify against text in the exact shape the formatter emits.
    probe = "<|im_start|>user\n1<|im_end|><|im_start|>assistant\n2<|im_end|>"
    try:
        ids = tokenizer.encode(probe)
        ok = USER_TOKEN_ID in ids and ASSISTANT_TOKEN_ID in ids and EOS_ID in ids
        print(f"[Rank {rank}] vocab probe ids={ids} turn_markers_found={ok}")
        if not ok:
            print(f"[Rank {rank}] !!! turn markers absent from encoded text -- "
                  f"the collator cannot find turn boundaries and every label "
                  f"will be masked (NaN loss)")
    except Exception as e:
        print(f"[Rank {rank}] vocab probe failed: {type(e).__name__}: {e}")


def load_fast_tokenizer(model_path, fallback, rank=0):
    """Load the checkpoint's tokenizer through the fast (tokenizers) backend.

    The checkpoint ships a 16-token WordLevel vocabulary in tokenizer.json, but
    its tokenizer_config.json declares `tokenizer_class: Qwen2Tokenizer`. Under
    transformers 5.x AutoTokenizer honours that and returns the *slow* BPE class,
    which rebuilds itself from vocab.json plus merges.txt. There is no merges.txt,
    so it cannot form the words `user` and `assistant` and drops them outright --
    no error, just missing tokens. The model was trained with those words present,
    so every prompt we sent was off-distribution.

    NVARC's transformers 4.55 defaulted to the fast class and never hit this.
    Loading PreTrainedTokenizerFast explicitly restores the intended behaviour.
    """
    probe = "<|im_start|>user\n1<|im_end|><|im_start|>assistant\n2<|im_end|>"
    try:
        from transformers import PreTrainedTokenizerFast
        fast = PreTrainedTokenizerFast.from_pretrained(model_path)
        if fast.pad_token_id is None:
            fast.pad_token = "<|endoftext|>"
        ids = fast.encode(probe)
        roles = {fast.convert_tokens_to_ids("user"),
                 fast.convert_tokens_to_ids("assistant")}
        if not roles.issubset(set(ids)):
            print(f"[Rank {rank}] fast tokenizer still drops role words {ids}; "
                  f"keeping the original")
            return fallback
        print(f"[Rank {rank}] fast tokenizer OK: {type(fast).__name__} probe={ids}")
        return fast
    except Exception as e:
        print(f"[Rank {rank}] fast tokenizer load failed: {type(e).__name__}: {e}")
        return fallback


def diagnose_tokenizer(tokenizer, rank=0):
    """Two checks that would otherwise fail silently and cost a submission.

    1. Compare unsloth's tokenizer against a plain AutoTokenizer load of the same
       directory. The checkpoint's tokenizer.json is byte-identical to the copy
       in our repo, which *does* emit the `user` / `assistant` ids when driven
       directly -- so if the plain load works and unsloth's does not, the damage
       is being done while loading, not by the file.
    2. Round-trip a grid: text -> ids -> text. Decoding is how candidate answers
       become grids again, and if newlines do not survive, every answer parses as
       a single row -- wrong shape, zero score, no error anywhere.
    """
    if rank != 0:
        return

    probe = "<|im_start|>user\n1<|im_end|><|im_start|>assistant\n2<|im_end|>"
    try:
        from transformers import AutoTokenizer
        plain = AutoTokenizer.from_pretrained(arc_config.model_path(),
                                              local_files_only=True)
        theirs, ours = plain.encode(probe), tokenizer.encode(probe)
        print(f"[Rank {rank}] AutoTokenizer : {theirs}")
        print(f"[Rank {rank}] unsloth       : {ours}")
        if theirs != ours:
            print(f"[Rank {rank}] !!! unsloth altered the tokenizer; the plain "
                  f"load is the correct behaviour")
    except Exception as e:
        print(f"[Rank {rank}] tokenizer comparison failed: {type(e).__name__}: {e}")

    grid_text = "123\n456\n789"
    try:
        ids = tokenizer.encode(grid_text)
        back = tokenizer.decode(ids)
        rows = [r for r in back.strip().split("\n") if r]
        ok = back.strip() == grid_text and len(rows) == 3
        print(f"[Rank {rank}] grid round-trip ids={ids} -> {back!r} rows={len(rows)} ok={ok}")
        if not ok:
            print(f"[Rank {rank}] !!! grids do not survive decode -- answers will "
                  f"parse with the wrong shape and score zero")
    except Exception as e:
        print(f"[Rank {rank}] grid round-trip failed: {type(e).__name__}: {e}")


class UnslothFixedTrainer(UnslothTrainer):

    # Issue https://github.com/unslothai/unsloth/issues/2435

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fixed compute_loss that handles Unsloth's view tensor issue"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        # 🔧 KEY FIX: Clone the loss tensor before in-place operations
        if hasattr(loss, "clone"):
            loss = loss.clone()  # Converts view tensor to independent tensor
        # Now safe for DDP gradient scaling
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        # Supervise only the assistant turns, i.e. the answer grids.
        #
        # NVARC located turn starts by the `user` / `assistant` token ids. On this
        # checkpoint those words are dropped during encoding -- the ids resolve
        # fine via convert_tokens_to_ids, but tokenizer.encode() never emits them
        # -- so no boundary was ever found, every label stayed -100, and the loss
        # became NaN over zero targets. Anchor on <|im_start|> / <|im_end|>, which
        # do survive, and find the newline that terminates the role header. That
        # works whether or not the role word is present.
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            ids = batch["input_ids"][i]
            labels = completion_labels(
                ids.cpu().numpy(), IM_START_ID, EOS_ID, NEWLINE_ID)
            batch["labels"][i] = torch.as_tensor(
                labels, dtype=batch["labels"].dtype, device=batch["labels"].device)
        return batch


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time) -> dict:

    n = logits.size(0)

    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)

    suffixes = defaultdict(list)

    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for t in ARC_TOKENS:
            score = nll[i, t].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0]) #[:5]
    
    while time.time() - start_time < arc_config.DFS_BUDGET_S and time.time() < end_time:

        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    batch_logits = outputs.logits.float().cpu().log_softmax(-1)
    result = []
    for logits, query_tokens, answer_tokens in zip(batch_logits, batch_query_tokens, batch_answer_tokens):
        query_length = len(query_tokens)
        answer_logits = logits[query_length-1:query_length-1+len(answer_tokens)]
        answer_score = answer_logits[torch.arange(len(answer_tokens)), answer_tokens].sum()
        result.append(-answer_score.item())
    return result


def worker(rank, queue, end_time):

    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=True,
        # Disable FSDP (use standard DDP)
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    max_seq_length = 8192

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=arc_config.model_path(),
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=max_seq_length,
    )

    tokenizer = load_fast_tokenizer(arc_config.model_path(), tokenizer, rank)

    init_vocab(tokenizer, rank)
    diagnose_tokenizer(tokenizer, rank)

    model = FastLanguageModel.get_peft_model(model, **peft_params)

    # NVARC cast every fp32 param to bf16 to save VRAM. Unsloth now deliberately
    # keeps the embedding adapters in fp32 ("Training embed_tokens in mixed
    # precision"), so stomping those to bf16 fights its own scheme. We have room
    # to spare on L4 (peak 8.6 of 22 GB), so leave them alone.
    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            if "embed_tokens" in name or "lm_head" in name:
                continue
            param.data = param.data.to(torch.bfloat16)

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )

    formatter = QwenFormatter(tokenizer=tokenizer)

    max_new_tokens = formatter.max_new_tokens()

    max_score = -np.log(0.2)

    arc_test_set = ArcDataset.from_file(arc_config.CHALLENGES)

    dir_outputs = arc_config.OUTPUT_DIR
    os.makedirs(dir_outputs, exist_ok=True)

    worker_start = time.time()
    num_done = 0

    while not queue.empty():

        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break

        start_time = time.time()
        
        torch.cuda.reset_peak_memory_stats()

        load_result = set_peft_model_state_dict(
            model,
            default_weights.copy(),
            adapter_name="default",
        )

        model = FastLanguageModel.for_training(model)

        puzzle_ds = arc_test_set.change_keys([key])

        train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)
        train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

        # One-off check that the completion-only collator still masks correctly.
        # If transformers changed DataCollatorForLanguageModeling under us, every
        # label could come back -100, and a loss over zero targets is NaN -- which
        # looks identical to an exploding-gradient NaN in the training stats.
        if num_done == 0:
            try:
                probe = [{"input_ids": tokenizer.encode(s["text"])}
                         for s in train_ds.as_list(formatter)[:2]]
                b = collator(probe)
                n_lab = int((b["labels"] != -100).sum())
                n_tok = int(b["labels"].numel())
                print(f"[Rank {rank}] collator check: {n_lab}/{n_tok} tokens "
                      f"supervised ({100 * n_lab / max(n_tok, 1):.1f}%)")
                if n_lab == 0:
                    print(f"[Rank {rank}] !!! collator masks EVERYTHING -- "
                          f"loss will be NaN regardless of the model")
            except Exception as e:
                print(f"[Rank {rank}] collator check failed: {type(e).__name__}: {e}")

        with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
            
            trainer = UnslothFixedTrainer(
                model=model,
                tokenizer=tokenizer,
                data_collator=collator,
                train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                args=UnslothTrainingArguments(**train_args),
            )

            stats = trainer.train()

            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

            del trainer

        model = FastLanguageModel.for_inference(model)
        
        gc.collect()
        torch.cuda.empty_cache()
            
        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for training")

        torch.cuda.reset_peak_memory_stats()
        
        print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")

        # A NaN loss means test-time fine-tuning learned nothing for this puzzle.
        # The run still completes and still emits candidates, so this would be
        # invisible without saying it out loud.
        loss = stats.metrics.get("train_loss") if hasattr(stats, "metrics") else None
        if loss is None or loss != loss:
            print(f"[Rank {rank}] !!! NaN training loss on {key} -- TTFT is not "
                  f"adapting the model; predictions are effectively untrained")

        puzzle_ds_multi = puzzle_ds.split_multi_replies()

        eval_ds = puzzle_ds_multi.augment(n=2, seed=2)
        eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)

        test_id_to_subkeys = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            test_id = subkey.split(".")[0].split("_")[1]
            test_id_to_subkeys[test_id].append(subkey)

        batches = []
        for test_id, subkeys in test_id_to_subkeys.items():
            # 0: permute x 2
            # 4: rot90.rot90.permute x 2
            batch = []
            for offset in [0, 4]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)
            # 2: permute.rot90 x 2
            # 6: rot90.rot90.rot90.permute x 2
            batch = []
            for offset in [2, 6]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)
        for test_id, subkeys in test_id_to_subkeys.items():
            # 8: transpose.permute x 2
            # 12: transpose.rot90.rot90.permute x 2
            batch = []
            for offset in [8, 12]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)
            # 10: transpose.rot90.permute x 2
            # 14: transpose.rot90.rot90.rot90.permute x 2
            batch = []
            for offset in [10, 14]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)

        with torch.inference_mode():
                
            known_scores = {}

            for subkeys in batches:

                spend_time = time.time() - start_time
                if spend_time > arc_config.PUZZLE_BUDGET_S or time.time() > end_time:
                    print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
                    break

                print(f"[Rank {rank}] decoding {subkeys}")

                tokens = []
                for subkey in subkeys:
                    data = eval_ds.get(subkey, formatter)
                    tokens.append(tokenizer.encode(data["input"]))

                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)

                for subkey_id, scored_beams in dfs_result:

                    subkey = subkeys[subkey_id]
                    bk = subkey.split(".")[0]
                    decoded_result = []

                    for beam_score, tokens in scored_beams:

                        array = formatter.convert_tokens_to_array(tokens)
                        if array is None:
                            continue

                        solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)

                        grid_id = (bk, tuple(map(tuple, solution)))

                        if grid_id in known_scores:
                            augmented_scores = known_scores[grid_id]
                        else:
                            print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                            aug_dataset = ArcDataset(
                                keys=[bk],
                                queries={bk: puzzle_ds_multi.queries.get(bk)},
                                replies={bk: [solution.tolist()]},
                            )
                            aug_dataset = aug_dataset.augment(seed=hash(bk) % 1024**2)
                            aug_dataset = aug_dataset.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
                            aug_queries = []
                            aug_answers = []
                            for augmented_sample in aug_dataset.as_list(formatter):
                                aug_queries.append(augmented_sample["input"])
                                aug_answers.append(augmented_sample["reply"])
                            augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model)
                            augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)
                            augmented_scores = augmented_scores1 + augmented_scores2
                            known_scores[grid_id] = augmented_scores
                        
                        decoded_result.append({
                            "beam_score": beam_score,
                            "score_aug": augmented_scores,
                            "solution": solution,
                        })

                    if len(decoded_result):
                        with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:
                            pickle.dump(decoded_result, f)

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")

        spend_time = time.time() - start_time
        num_done += 1

        # Pace check. 240 tasks over NUM_WORKERS must clear the deadline; at
        # NVARC's 1200s ceiling they would not, so watch the running average.
        avg = (time.time() - worker_start) / num_done
        affordable = max(0, int((end_time - time.time()) / avg)) if avg > 0 else 0
        print(
            f"[Rank {rank}] finished {key} in {spend_time:.1f}s "
            f"| done={num_done} avg={avg:.1f}s "
            f"| {affordable} more fit before deadline"
        )