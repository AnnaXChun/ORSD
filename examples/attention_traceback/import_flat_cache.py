"""Convert an existing unpadded token/mask/message cache; no environment calls.

Required per row: prompt_ids, response_ids, response_mask, messages, benchmark,
task_id, trajectory_id, trajectory_reward, success, done_reason, step_rewards,
model_fingerprint, tokenizer_fingerprint. Fingerprints may be supplied from the
original run manifest using --provenance (never guessed from a new checkpoint).
"""
import argparse
import json
from pathlib import Path


def main():
    from transformers import AutoTokenizer
    from verl.experimental.attention_traceback.capture import action_span, observation_end, tokenizer_fingerprint, checkpoint_fingerprint
    from verl.experimental.attention_traceback.replay import validate_record
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='JSONL, unpadded veRL token cache')
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--provenance', help='original run manifest with model_fingerprint/tokenizer_fingerprint')
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    provenance = json.loads(Path(args.provenance).read_text()) if args.provenance else {}
    expected_model = checkpoint_fingerprint(args.model)
    expected_tokenizer = tokenizer_fingerprint(tokenizer)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    converted = []
    for line in Path(args.input).read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        for key in ('benchmark', 'task_id', 'trajectory_id', 'trajectory_reward', 'success', 'done_reason', 'step_rewards', 'messages'):
            if key not in record:
                raise ValueError(f'Missing original metadata: {key}; text-only traces cannot establish exact replay')
        for key, expected in (('model_fingerprint', expected_model), ('tokenizer_fingerprint', expected_tokenizer)):
            if record.get(key, provenance.get(key)) != expected:
                raise ValueError(f'Missing/mismatched original {key}; recover the original run manifest')
            record[key] = expected
        prefix, response, mask = record.pop('prompt_ids'), record.pop('response_ids'), record.pop('response_mask')
        if len(response) != len(mask) or any(m not in (0, 1) for m in mask) or not mask or mask[0] != 1:
            raise ValueError('Expected unpadded response IDs and binary generation mask starting with an assistant turn')
        pairs = []
        messages = record['messages']
        for i, message in enumerate(messages):
            if message['role'] == 'assistant':
                if i + 1 == len(messages) or messages[i + 1]['role'] != 'user':
                    raise ValueError('Missing resulting observation for an assistant decision')
                pairs.append((message['content'], messages[i + 1]['content']))
        turns = []
        i = 0
        kwargs = record.get('chat_template_kwargs', provenance.get('chat_template_kwargs', {}))
        while i < len(response):
            a = i
            while i < len(response) and mask[i] == 1:
                i += 1
            b = i
            while i < len(response) and mask[i] == 0:
                i += 1
            c = i
            if a == b or b == c or len(turns) >= len(pairs):
                raise ValueError('Cache is incomplete or mask/message turns disagree')
            sampled = response[a:b]
            s, e, action, selection = action_span(tokenizer, sampled)
            assistant, observation = pairs[len(turns)]
            decoded = tokenizer.decode(sampled, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if decoded.strip() != assistant.strip():
                raise ValueError('Stored assistant text differs from sampled token IDs')
            content_end = len(prefix) + b + observation_end(tokenizer, response[b:c], kwargs)
            t = len(turns)
            turns.append({'turn_idx': t, 'action_text': action, 'response_text': assistant,
                'observation_text': observation, 'action_token_start': len(prefix) + a + s,
                'action_token_end': len(prefix) + a + e, 'action_token_ids': sampled[s:e],
                'action_token_logps': [], 'turn_token_start': len(prefix) + a,
                'turn_token_end': content_end, 'response_token_end': len(prefix) + b,
                'observation_token_start': len(prefix) + b, 'observation_token_end': content_end,
                'token_selection': selection, 'reward': record['step_rewards'][t]})
        if len(turns) != len(pairs) or len(turns) != len(record['step_rewards']):
            raise ValueError('Step count differs between rewards, messages and tokens')
        record.update(schema_version=1, input_ids=prefix + response, turns=turns,
                      model_path=str(Path(args.model).resolve()), grpo_advantage=None,
                      chat_template_kwargs=kwargs, cache_provenance='original_flat_token_cache')
        validate_record(record)
        converted.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        for record in converted:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
