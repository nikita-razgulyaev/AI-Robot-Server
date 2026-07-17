import requests

key = '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'

payload = {
    'model': 'gpt-4o-mini',
    'messages': [
        {'role': 'system', 'content': 'Ты — Сорен, амбарная сова.'},
        {'role': 'user', 'content': 'Привет!'}
    ],
    'temperature': 0.6,
    'max_tokens': 256
}

resp = requests.post(
    'https://models.inference.ai.azure.com/chat/completions',
    headers={'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'},
    json=payload,
    timeout=30
)
print(f'Status: {resp.status_code}')
print(f'Body: {resp.text[:500]}')