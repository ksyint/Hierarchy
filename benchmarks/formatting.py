def format_pair(record, hard, explicit):
    rejected_key = 'rejected_hard' if hard else 'rejected_easy'
    prompt, chosen, rejected = record['prompt'], record['chosen'], record[rejected_key]
    if explicit:
        def add_thinking(answer, key):
            thought = record.get(key + '_thinking', '')
            return f'[THINKING]{thought}[/THINKING]\n{answer}' if thought else answer
        chosen = add_thinking(chosen, 'chosen')
        rejected = add_thinking(rejected, rejected_key)
    else:
        prompt += '\n[WITHOUT_THINKING]\n'
    return prompt, chosen, rejected
